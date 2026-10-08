#!/opt/pyenv/versions/fednpg/bin/python
"""
Progress/health check for runs in results/, for use while a sweep is going.

    ./sweep_status.py                 # every run touched in the last 24h
    ./sweep_status.py hopper-bare     # only runs whose id contains this
    ./sweep_status.py --all           # every run on disk
    ./sweep_status.py -w              # refresh every 30s

Answers the three questions a running sweep actually raises: is the process
still alive, how far along is it, and is it producing something worth waiting
for. The last one matters most -- a run can be perfectly healthy by every
process metric and still be heading somewhere useless, which no "is it running"
check would catch.

ALIVE is resolved by matching the run id against the --run-id argument of the
live processes, not by file mtime: metrics.csv is appended every iteration, so
a stalled run's file still looks recently touched for a while, and a run
between its last write and its checkpoint save looks idle when it is not.

STATE column:
  done     metadata.json has completed_at
  running  a process holds this --run-id
  queued   incomplete, but the --sweep target will redo it; nothing is wrong
  DEAD     no completed_at and no process  <- crashed or was killed; check the log
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def live_run_ids() -> set[str]:
    """Run ids currently held by a training process, read off its command line."""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return set()
    ids = set()
    for line in out.splitlines():
        if "--run-id" not in line:
            continue
        parts = line.split()
        try:
            ids.add(parts[parts.index("--run-id") + 1])
        except (ValueError, IndexError):
            pass
    return ids


def tail_float(rows: list[dict], col: str, n: int = 5) -> float:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
    return sum(vals[-n:]) / len(vals[-n:]) if vals else float("nan")


def measured_rates() -> dict[tuple[str, int], float]:
    """
    Seconds per iteration, keyed by (env, num_agents), measured from every
    COMPLETED run on disk. Used to cost runs a sweep has not started yet.

    Measured rather than assumed because cost is not linear in num_agents and
    not comparable across environments: Hopper on CPU runs 0.6/0.9/1.3/1.9 h at
    N=1/2/4/8 for 2000 iterations, so a naive "N times longer" estimate is off
    by more than 3x at N=8.
    """
    rates: dict[tuple[str, int], list[float]] = {}
    for d in RESULTS.iterdir():
        meta_p = d / "metadata.json"
        if not d.is_dir() or not meta_p.exists():
            continue
        try:
            m = json.loads(meta_p.read_text())
        except json.JSONDecodeError:
            continue
        if not m.get("completed_at") or not m.get("iterations"):
            continue
        try:
            secs = (
                datetime.fromisoformat(m["completed_at"]) - datetime.fromisoformat(m["started_at"])
            ).total_seconds()
        except (KeyError, ValueError):
            continue
        key = (str(m.get("env")), int(m.get("num_agents", 0)))
        rates.setdefault(key, []).append(secs / int(m["iterations"]))
    # Median, not mean: a single run that shared the machine with fifteen others
    # would otherwise drag the whole estimate.
    return {k: sorted(v)[len(v) // 2] for k, v in rates.items()}


def expected_runs(target: str, seeds: list[str], agents: list[int]) -> list[tuple[str, str, int, int]]:
    """(run_id, env, num_agents, iterations) for every run a sweep target will do."""
    try:
        from common.hyperparams import DEFAULT_HYPERPARAMS as HP

        iters = {e: HP[e].iterations for e in ("swimmer", "hopper", "humanoid")}
    except Exception:  # package not importable from here; Table 4 values
        iters = {"swimmer": 1000, "hopper": 2000, "humanoid": 3000}

    if target.startswith("final-"):
        env = target[len("final-"):]
        if env not in iters:
            raise SystemExit(f"unknown sweep target: {target}")
        return [
            (f"final-{env}-N{n}-s{s}", env, n, iters[env])
            for s in seeds
            for n in agents
        ]
    if target == "hopper":
        return [
            (f"hopper-bare-N{n}-s{s}", "hopper", n, iters["hopper"])
            for s in seeds
            for n in agents
        ]
    if target == "isolate":
        names = ["linesearch", "obsnorm", "rewnorm", "advnorm", "logstdclamp"]
        return [(f"hopper-iso-{x}", "hopper", 8, iters["hopper"]) for x in names]
    if target == "crosscheck":
        return [
            (f"{env}-bare-N{n}-s0", env, n, iters[env])
            for env in ("swimmer", "humanoid")
            for n in (1, 8)
        ]
    raise SystemExit(f"unknown sweep target: {target}")


def sweep_eta(target: str, seeds: list[str], agents: list[int], wave: int, live: set[str]) -> str:
    """
    Wall-clock remaining for a whole sweep target, including runs not started.

    Models what run_bare_sweep.sh actually does: at most `wave` trainings in
    flight, the next one starting as a slot frees. Greedy longest-first onto
    the earliest-free slot, which is how the real queue behaves closely enough
    for an estimate -- it is a planning number, not a guarantee, and it assumes
    the machine stays as loaded as it is now.
    """
    rates = measured_rates()
    pending: list[float] = []   # seconds of work left, per unfinished run
    n_done = n_run = n_todo = 0

    for run_id, env, n, total in expected_runs(target, seeds, agents):
        d = RESULTS / run_id
        meta_p = d / "metadata.json"
        meta = {}
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text())
            except json.JSONDecodeError:
                pass
        if meta.get("completed_at"):
            n_done += 1
            continue

        rate = rates.get((env, n))
        if rate is None:  # no comparable run finished yet; scale from any N in this env
            same_env = [r for (e, _), r in rates.items() if e == env]
            rate = (sum(same_env) / len(same_env)) if same_env else 3.5

        csv_p = d / "metrics.csv"
        done_it = 0
        if csv_p.exists():
            try:
                done_it = max(0, sum(1 for _ in csv_p.open()) - 1)
            except OSError:
                done_it = 0
        # A live run's own observed rate beats the historical one.
        if run_id in live and done_it > 5 and meta.get("started_at"):
            try:
                el = (datetime.now(timezone.utc) - datetime.fromisoformat(meta["started_at"])).total_seconds()
                rate = el / done_it
            except ValueError:
                pass
        if run_id in live:
            n_run += 1
        else:
            n_todo += 1
        pending.append(max(0.0, (total - done_it) * rate))

    if not pending:
        return f"sweep '{target}': all {n_done} run(s) complete"

    slots = [0.0] * max(1, wave)
    for work in sorted(pending, reverse=True):
        i = min(range(len(slots)), key=lambda j: slots[j])
        slots[i] += work
    makespan = max(slots)
    finish = datetime.now().astimezone() + timedelta(seconds=makespan)
    serial = sum(pending) / 3600

    return (
        f"sweep '{target}': {n_done} done, {n_run} running, {n_todo} queued"
        f"  |  {makespan/3600:.1f} h remaining at wave={wave}"
        f"  ->  finishes ~{finish:%a %H:%M}"
        f"   ({serial:.1f} h of work total)"
    )


def summarize(d: Path, live: set[str]) -> dict | None:
    meta_p, csv_p = d / "metadata.json", d / "metrics.csv"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return None

    rows: list[dict] = []
    if csv_p.exists():
        try:
            rows = list(csv.DictReader(csv_p.open()))
        except Exception:
            rows = []

    total = int(meta.get("iterations") or 0)
    done_n = len(rows)
    completed = bool(meta.get("completed_at"))
    state = "done" if completed else ("running" if d.name in live else "DEAD")

    # Rate and ETA from the run's own elapsed time, not from file timestamps.
    start = meta.get("started_at")
    eta = ""
    done_at = ""
    if start and not completed and done_n > 1:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds()
        per_it = elapsed / done_n
        remaining = (total - done_n) * per_it
        if remaining > 0:
            eta = f"{remaining / 3600:.1f}h"
            done_at = (datetime.now().astimezone() + timedelta(seconds=remaining)).strftime("%H:%M")

    return {
        "id": d.name,
        "state": state,
        "pct": (100.0 * done_n / total) if total else float("nan"),
        "it": f"{done_n}/{total}" if total else str(done_n),
        "eta": eta,
        "done_at": done_at,
        "train": tail_float(rows, "ep_rew_mean"),
        "eval": tail_float(rows, "eval_ep_rew_mean"),
        "eplen": tail_float(rows, "eval_ep_len_mean"),
        "std": tail_float(rows, "mean_action_std"),
        "mtime": csv_p.stat().st_mtime if csv_p.exists() else 0.0,
    }


def render(pattern: str | None, show_all: bool, args=None) -> str:
    live = live_run_ids()
    # Runs a sweep still intends to do. Without this, a run killed mid-flight
    # and now waiting its turn for a re-run shows as DEAD, which sends you off
    # reading logs for a failure that is not there -- the sweep script skips
    # only runs with completed_at, so every incomplete one WILL be redone.
    queued: set[str] = set()
    if args is not None and getattr(args, "sweep", None):
        try:
            queued = {
                rid
                for rid, *_ in expected_runs(
                    args.sweep, args.seeds.split(), [int(x) for x in args.agents.split()]
                )
            } - live
        except SystemExit:
            queued = set()
    cutoff = time.time() - 24 * 3600
    rows = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        if pattern and pattern not in d.name:
            continue
        s = summarize(d, live)
        if s is None:
            continue
        if s["state"] == "DEAD" and s["id"] in queued:
            s["state"] = "queued"
        if not show_all and not pattern and s["mtime"] < cutoff and s["state"] != "running":
            continue
        rows.append(s)
    rows.sort(key=lambda r: (r["state"] != "running", r["state"] != "DEAD",
                             r["state"] != "queued", r["id"]))

    out = [
        f"{'run':<30}{'state':>8}{'iters':>12}{'%':>6}{'eta':>7}"
        f"{'train':>8}{'eval':>8}{'eplen':>7}{'std':>7}",
        "-" * 100,
    ]
    for r in rows:
        def f(v, w, p=0):
            return f"{v:>{w}.{p}f}" if v == v else f"{'-':>{w}}"  # v==v filters NaN
        out.append(
            f"{r['id']:<30}{r['state']:>8}{r['it']:>12}{f(r['pct'],6)}{r['eta']:>7}{r['done_at']:>7}"
            f"{f(r['train'],8)}{f(r['eval'],8)}{f(r['eplen'],7)}{f(r['std'],7,3)}"
        )
    n_dead = sum(1 for r in rows if r["state"] == "DEAD")
    out.append("")
    out.append(
        f"{len(rows)} run(s): {sum(1 for r in rows if r['state']=='running')} running, "
        f"{sum(1 for r in rows if r['state']=='done')} done, {n_dead} dead"
        + ("   <- inspect results/<id>.log" if n_dead else "")
    )
    if args is not None and args.sweep:
        out.append("")
        out.append(
            sweep_eta(args.sweep, args.seeds.split(), [int(x) for x in args.agents.split()],
                      args.wave, live)
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern", nargs="?", help="substring of the run id to filter on")
    p.add_argument("--all", action="store_true", help="include runs older than 24h")
    p.add_argument("-w", "--watch", action="store_true", help="refresh until interrupted (see --interval)")
    p.add_argument("--interval", type=float, default=15.0, help="seconds between refreshes (default: 15)")
    p.add_argument("--sweep", choices=["final-hopper", "final-swimmer", "final-humanoid", "hopper", "isolate", "crosscheck"],
                   help="also report wall-clock remaining for a whole run_bare_sweep.sh target, "
                        "including runs it has not started yet")
    p.add_argument("--seeds", default="0 1 2 3 4", help="seeds the sweep will cover (default: run_bare_sweep.sh's)")
    p.add_argument("--agents", default="1 2 4 8", help="agent counts the sweep will cover")
    p.add_argument("--wave", type=int, default=4, help="concurrent trainings, matching run_bare_sweep.sh's WAVE")
    a = p.parse_args()
    if a.sweep and a.sweep.startswith("final-") and "--wave" not in sys.argv:
        a.wave = len(a.seeds.split())  # final_sweep.sh: one pane per seed

    if not a.watch:
        print(render(a.pattern, a.all, a))
        return

    # Repaint only when stdout is a terminal. Emitting the clear-screen escape
    # into a pipe or a log file writes literal control bytes and scrolls
    # nothing, which is what made this look frozen when piped; and without an
    # explicit flush, a non-TTY stdout is block-buffered, so nothing appeared
    # at all until the process was killed -- by which point the buffer was
    # discarded.
    tty = sys.stdout.isatty()
    try:
        while True:
            if tty:
                sys.stdout.write("\033[H\033[J")  # home, then clear to end
            print(
                f"{datetime.now():%F %T}   refreshing every {a.interval:g}s   (ctrl-C to stop)",
                flush=True,
            )
            print(render(a.pattern, a.all, a), flush=True)
            # Sleep in short slices so ctrl-C lands immediately rather than
            # after the remainder of a long interval.
            waited = 0.0
            while waited < a.interval:
                time.sleep(min(0.5, a.interval - waited))
                waited += 0.5
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
