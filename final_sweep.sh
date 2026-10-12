#!/usr/bin/env bash
# Hopper sweeps, in the reduced configuration the published Hopper results use:
# no reward, advantage or observation normalization, no line search and no
# log-std clamp (Swimmer and Humanoid use the standard settings; see
# run_sweep.sh). Each target opens one tmux session with live progress bars.
#
#   ./final_sweep.sh hopper             FedNPG-ADMM, 5 seeds x N = 8, 4, 2, 1
#   ./final_sweep.sh hopper-baselines   standard-average FedNPG + FedPPO, N=8
#   ./final_sweep.sh hopper-quant       stochastic quantization, 6/8 bits, N=2/4/8
#   DRY_RUN=1 ./final_sweep.sh <target> print the commands, start nothing
#
#   tmux attach -t final-hopper         (inside tmux: tmux switch-client -t ...)
#   ./sweep_status.py --sweep final-hopper   progress table + ETA
#
# Run ids start with "final-hopper-". Resumable: a run whose metadata.json has
# completed_at is skipped, so re-running continues where it stopped.
# Requires the package CLIs on PATH (activate the environment first).
set -u
ENV_NAME="${1:?usage: final_sweep.sh hopper|hopper-baselines|hopper-quant}"
if [ "$ENV_NAME" = all ]; then
  rc=0
  for e in hopper; do "$0" "$e" || rc=1; done
  exit $rc
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$(command -v fedrl-train-admm)}"
SEEDS="${SEEDS:-0 1 2 3 4}"
AGENTS="${AGENTS:-8 4 2 1}"
# Device: CPU for every environment. Humanoid was tried on the GPU on
# 2026-10-07 and was slower than CPU, matching the earlier measurement in
# run_sweep.sh (Humanoid N=8: 5.47 s/it CPU vs 6.59 GPU). The policies are
# small enough that MuJoCo stepping and per-step launch overhead dominate.
# Override with DEVICE=cuda:0 if that ever changes.
DEVICE="${DEVICE:-cpu}"
COMMON="--device $DEVICE --eval-interval 25 --eval-episodes 10 --save-checkpoint --on-invalid-step skip"

case "$ENV_NAME" in
  hopper)
    # Measured as test-hopper-iso-orthoinit: 3678 at iteration 2000, 1000-step
    # episodes, no decay (default config: 3854 peak -> 1870).
    CONFIG="--no-line-search --no-obs-normalization --no-reward-normalization \
--no-advantage-normalization --log-std-min -20 --log-std-max 20"
    ;;
  swimmer|humanoid)
    echo "$ENV_NAME uses the standard configuration: run ./run_sweep.sh $ENV_NAME instead." >&2
    exit 2
    ;;
  hopper-baselines|hopper-quant) CONFIG="" ;;   # handled by their own blocks below
  *) echo "unknown env: $ENV_NAME" >&2; exit 2 ;;
esac
# Iterations are NOT passed: each environment falls back to its Table 4 K
# (Swimmer 1000, Hopper 2000, Humanoid 3000).

run_id() { echo "final-${ENV_NAME}-N$1-s$2"; }
done_already() { [ -f "$REPO/results/$1/metadata.json" ] && grep -q '"completed_at"' "$REPO/results/$1/metadata.json"; }
# --- Hopper quantization grid -------------------------------------------------
# Stochastic quantization of the agent-to-server uplink on Hopper (6 and 8
# bits, N=2/4/8, 5 seeds), in the same reduced configuration as above. N=1 is
# not quantized: one agent has no uplink, and final-hopper-N1-* covers it.
# All 30 runs in parallel, one process each, via scripts/run_grid.py, which
# names them hopper-N<n>-s<seed>-q<bits>; they go to results/hopper-quant/.
# One tmux session; one window per (N, bits), five seed panes in each.
if [ "$ENV_NAME" = hopper-quant ]; then
  QDIR="$REPO/results/hopper-quant"
  HOPPER_CFG="--no-line-search --no-obs-normalization --no-reward-normalization \
--no-advantage-normalization --log-std-min -20 --log-std-max 20"
  qcmd() {  # $1 = N, $2 = bits, $3 = seed
    echo "cd $REPO && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHON:-$(command -v python)} scripts/run_grid.py --env hopper --seed-start $3 --num-seeds 1 --agents $1 --bits $2 --scheme stochastic --results-path $QDIR $HOPPER_CFG --device cpu --eval-interval 25 --eval-episodes 10 --save-checkpoint"
  }
  if [ "${DRY_RUN:-0}" = 1 ]; then
    for n in 8 4 2; do for b in 6 8; do for s in $SEEDS; do qcmd $n $b $s; echo; done; done; done; exit 0
  fi
  SESSION=final-hopper-quant
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[exists] $SESSION -- left alone (attach: tmux attach -t $SESSION)"; exit 0
  fi
  first_window=1
  for n in 8 4 2; do
    for b in 6 8; do
      win="N$n-q$b"
      if [ $first_window -eq 1 ]; then
        tmux new-session -d -s "$SESSION" -n "$win" -x 240 -y 70
        tmux set-option -t "$SESSION" pane-border-status top
        tmux set-option -t "$SESSION" pane-border-format " #{@name} "
        first_window=0
      else
        tmux new-window -t "$SESSION" -n "$win"
      fi
      first_pane=1
      for s in $SEEDS; do
        if [ -f "$QDIR/hopper-N$n-s$s-q$b/metadata.json" ] && grep -q '"completed_at"' "$QDIR/hopper-N$n-s$s-q$b/metadata.json"; then
          echo "[skip] hopper-N$n-s$s-q$b (already complete)"; continue
        fi
        if [ $first_pane -eq 0 ]; then
          tmux split-window -t "$SESSION:$win"; tmux select-layout -t "$SESSION:$win" tiled
        fi
        first_pane=0
        tmux set-option -p -t "$SESSION:$win" @name "hopper N=$n ${b}-bit seed $s"
        tmux send-keys -t "$SESSION:$win" "$(qcmd $n $b $s)" C-m
      done
      tmux select-layout -t "$SESSION:$win" tiled
    done
  done
  echo "started $SESSION: 6 windows (N=8,4,2 x 6,8 bits), 5 seed panes each -> $QDIR"
  echo "  watch:  tmux attach -t $SESSION      (Ctrl-b w lists the windows, Ctrl-b n/p switches)"
  echo "  table:  ./sweep_status.py --all hopper-N"
  exit 0
fi

# --- Hopper baselines (Figure 3): standard-average FedNPG and FedPPO ---------
# Same settings as FedNPG-ADMM's Hopper runs, so the comparison measures the
# algorithms rather than the settings. N=8, as in the paper's Figure 3.
# One tmux session, one pane per (algorithm, seed): all 10 in parallel.
# Swimmer and Humanoid baselines: ./run_sweep.sh base-swimmer|base-humanoid.
if [ "$ENV_NAME" = hopper-baselines ]; then
  BBIN="${BBIN:-$(command -v fedrl-train-baseline)}"
  HOPPER_CFG="--no-line-search --no-obs-normalization --no-reward-normalization \
--no-advantage-normalization --log-std-min -20 --log-std-max 20"
  bcmd() {  # $1 = algo, $2 = seed
    echo "cd $REPO && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $BBIN --algo $1 --env hopper --num-agents 8 --seed $2 $HOPPER_CFG --device cpu --eval-interval 25 --eval-episodes 10 --save-checkpoint --run-id final-hopper-$1-N8-s$2"
  }
  if [ "${DRY_RUN:-0}" = 1 ]; then
    for a in fednpg-standard fedppo; do for s in $SEEDS; do bcmd $a $s; echo; done; done; exit 0
  fi
  SESSION=final-hopper-baselines
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[exists] $SESSION -- left alone (attach: tmux attach -t $SESSION)"; exit 0
  fi
  tmux new-session -d -s "$SESSION" -x 240 -y 70
  tmux set-option -t "$SESSION" pane-border-status top
  tmux set-option -t "$SESSION" pane-border-format " #{@name} "
  first=1
  for a in fednpg-standard fedppo; do
    for s in $SEEDS; do
      id="final-hopper-$a-N8-s$s"
      if done_already "$id"; then echo "[skip] $id (already complete)"; continue; fi
      if [ $first -eq 0 ]; then
        tmux split-window -t "$SESSION"; tmux select-layout -t "$SESSION" tiled
      fi
      first=0
      tmux set-option -p -t "$SESSION" @name "$a seed $s"
      tmux send-keys -t "$SESSION" "$(bcmd $a $s)" C-m
    done
  done
  tmux select-layout -t "$SESSION" tiled
  echo "started $SESSION: fednpg-standard + fedppo, seeds [$SEEDS], N=8"
  echo "  watch:  tmux attach -t $SESSION"
  echo "  table:  ./sweep_status.py final-hopper-fed"
  exit 0
fi

cmd_for() {  # $1 = N, $2 = seed
  echo "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $BIN --env $ENV_NAME --num-agents $1 --seed $2 $CONFIG $COMMON --run-id $(run_id "$1" "$2")"
}

# --- worker: one seed's four runs, in the foreground of a tmux pane ---------
if [ "${2:-}" = "--worker" ]; then
  seed="${3:?--worker needs a seed}"
  cd "$REPO"
  for n in $AGENTS; do
    id="$(run_id "$n" "$seed")"
    if done_already "$id"; then echo "[skip] $id (already complete)"; continue; fi
    echo "=== $(date '+%F %T')  start $id"
    eval "$(cmd_for "$n" "$seed")"
  done
  echo "=== $(date '+%F %T')  seed $seed finished all runs"
  exit 0
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  for s in $SEEDS; do for n in $AGENTS; do cmd_for "$n" "$s"; echo; done; done
  exit 0
fi

SESSION="final-${ENV_NAME}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[exists] $SESSION -- left alone (attach: tmux attach -t $SESSION)"
  exit 0
fi

# Explicit size: a detached session defaults to 80x24, which runs out of room
# after a few splits. Panes resize to your terminal on attach anyway.
tmux new-session -d -s "$SESSION" -x 240 -y 70
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format " #{@name} "
first=1
for s in $SEEDS; do
  if [ $first -eq 0 ]; then
    tmux split-window -t "$SESSION"
    tmux select-layout -t "$SESSION" tiled   # rebalance so the next split has room
  fi
  first=0
  # Pane label via a user option: the shell can't overwrite it like a title.
  tmux set-option -p -t "$SESSION" @name "${ENV_NAME} seed ${s}  (N=${AGENTS// /,})"
  tmux send-keys -t "$SESSION" "$REPO/final_sweep.sh $ENV_NAME --worker $s" C-m
done
tmux select-layout -t "$SESSION" tiled

echo "started $SESSION: 5 panes, seeds [$SEEDS], each N=${AGENTS// /,} back to back"
echo "  watch:  tmux attach -t $SESSION"
echo "  table:  ./sweep_status.py --sweep $SESSION"
