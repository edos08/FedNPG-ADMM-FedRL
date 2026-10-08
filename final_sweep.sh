#!/usr/bin/env bash
# Final sweep for the report: one environment per invocation, 5 seeds x
# num_agents 1/2/4/8, Table 4 iterations, live tqdm progress bars in tmux.
#
#   ./final_sweep.sh all             # all three environments x 5 seeds = 15 sessions
#   ./final_sweep.sh hopper          # FedNPG-ADMM on Hopper
#   ./final_sweep.sh hopper-baselines   # Figure 3 baselines on Hopper, same settings
#
# One tmux session per environment (final-hopper, final-swimmer,
# final-humanoid), each with 5 panes, one per seed. Each pane runs that seed's
# four agent counts back to back (N=8, 4, 2, 1) and shows the live progress
# bar of the current run.
#
#   tmux attach -t final-hopper      # (inside tmux: tmux switch-client -t final-hopper)
#   Ctrl-b z                         # zoom one pane in/out;  Ctrl-b s  switch session
#   ./sweep_status.py --sweep final-hopper     # table + whole-sweep ETA
#   DRY_RUN=1 ./final_sweep.sh hopper          # print the 20 commands, start nothing
#
# At most 5 trainings per environment are in flight (one per seed), 15 with
# all three environments -- which this 46-core machine carries alongside other
# users' jobs.
#
# Resumable: a run whose metadata.json has completed_at is skipped, so
# re-running this after a crash or a reboot continues where it stopped.
# Run ids start with "final-" so nothing here mixes with earlier
# experiments' results (e.g. hopper-N8-s0 is the old default configuration).
#
# Per-environment configuration -- why each differs is in
# docs/hopper_agent_scaling_findings.md. Short version: reward normalization
# and advantage normalization each cause Hopper's late collapse at high N,
# so both are off everywhere; observation normalization is essential on
# Swimmer and Humanoid and hurts Hopper, so it is on for those two only.
set -u
ENV_NAME="${1:?usage: final_sweep.sh all|hopper|hopper-baselines|swimmer|humanoid [--worker SEED]}"
if [ "$ENV_NAME" = all ]; then
  rc=0
  for e in hopper; do "$0" "$e" || rc=1; done
  exit $rc
fi
REPO=/mnt/edoardo/FedNPG-ADMM-FedRL
BIN="${BIN:-/opt/pyenv/versions/fednpg/bin/fedrl-train-admm}"
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
    # Not re-run: the original configuration already reproduces the paper's
    # agent-count ordering on these two, and their 5-seed results are on disk
    # as {swimmer,humanoid}-N{1,2,4,8}-s{0..4}. See
    # docs/hopper_agent_scaling_findings.md section 4.
    echo "$ENV_NAME uses the original configuration; its report runs already exist" >&2
    echo "(results/${ENV_NAME}-N{1,2,4,8}-s{0..4}). Nothing to launch." >&2
    exit 0
    ;;
  hopper-baselines) CONFIG="" ;;   # handled by its own block below
  *) echo "unknown env: $ENV_NAME" >&2; exit 2 ;;
esac
# Iterations are NOT passed: each environment falls back to its Table 4 K
# (Swimmer 1000, Hopper 2000, Humanoid 3000).

run_id() { echo "final-${ENV_NAME}-N$1-s$2"; }
done_already() { [ -f "$REPO/results/$1/metadata.json" ] && grep -q '"completed_at"' "$REPO/results/$1/metadata.json"; }
# --- Hopper baselines (Figure 3): standard-average FedNPG and FedPPO ---------
# Same practical settings as FedNPG-ADMM's Hopper runs, so the comparison
# measures the algorithms rather than the settings. Both baselines showed the
# same late decay under the original settings (standard FedNPG -10..-19% from
# peak, FedPPO -22..-49%, 5 seeds each). N=8 only, as in the paper's Figure 3.
# One tmux session, one pane per (algorithm, seed): all 10 in parallel.
# Swimmer and Humanoid baselines are NOT re-run: their existing runs used the
# original configuration, the same one their FedNPG-ADMM runs use.
if [ "$ENV_NAME" = hopper-baselines ]; then
  BBIN="${BBIN:-/opt/pyenv/versions/fednpg/bin/fedrl-train-baseline}"
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
