#!/usr/bin/env bash
# Tonight's test runs: 12 trainings in parallel, one tmux pane each, so every
# run shows its own live tqdm progress bar.
#
#   ./launch_tests.sh            start everything (detached)
#   tmux attach -t tests         watch the bars
#                                  Ctrl-b z  zoom one pane in/out
#                                  Ctrl-b d  detach (runs keep going)
#   ./sweep_status.py test-      table view: progress, ETA, latest eval
#   DRY_RUN=1 ./launch_tests.sh  print the 12 commands without starting anything
#
# What the three groups answer. Together they decide tomorrow's ONE
# configuration for the final sweep on all three environments.
#
#   test-hopper-iso-*     "bare" Algorithm 1 with exactly ONE of the six
#                         practical settings restored. Bare alone climbs to
#                         3458 at num_agents=8 without collapsing
#                         (hopper-bare-N8); the default config, with all six
#                         on, peaks at 3854 and decays to 1870
#                         (hopper-N8-s0). A run here that collapses late names
#                         the component responsible.
#   test-*-bare-*         bare on swimmer/humanoid: does removing everything
#                         break the two environments that already reproduce
#                         the paper?
#   test-*-iso-obsnorm    bare + observation normalization on swimmer and
#                         humanoid. Observation normalization is the one
#                         setting already known to matter for Swimmer, so
#                         "bare + obsnorm" is the likeliest single
#                         configuration that works on all three;
#                         test-hopper-iso-obsnorm covers its Hopper side.
#
# Humanoid runs stop at 1500 of Table 4's 3000 iterations so the batch lands
# by morning. That is enough to decide: the default config's Humanoid N=8
# curve makes its big jump between iterations 500 and 1000 (643 -> 5841) and
# is at 6230 by 1500, so a matched-iteration comparison against
# humanoid-N8-s0 settles whether a configuration breaks it.
#
# Every run id starts with "test-" so nothing here can collide with, or be
# skipped as already-done by, tomorrow's final sweep.
#
# Every run spells its flags out in full. An earlier isolation attempt reused
# a shared $BARE string that still contained --no-line-search, so its
# "plus line search" run was a byte-identical duplicate of bare and the
# experiment silently never happened.
set -u

REPO=/mnt/edoardo/FedNPG-ADMM-FedRL
BIN="${BIN:-/opt/pyenv/versions/fednpg/bin/fedrl-train-admm}"
SESSION="${SESSION:-tests}"
COMMON="--seed 0 --device cpu --eval-interval 25 --eval-episodes 10 --save-checkpoint"

NAMES=()
CMDS=()
run() {  # $1 = run id, rest = training flags
  local name="$1"; shift
  NAMES+=("$name")
  CMDS+=("cd $REPO && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $BIN $* $COMMON --run-id $name")
}

# --- Hopper isolation: bare + exactly one setting restored, num_agents=8 ------
# The full bare set, for reference (each run below omits exactly one item):
#   --no-line-search --no-obs-normalization --no-reward-normalization
#   --no-advantage-normalization --no-ortho-init --log-std-min -20 --log-std-max 20

run test-hopper-iso-linesearch --env hopper --num-agents 8 --iterations 2000 \
  --no-obs-normalization --no-reward-normalization --no-advantage-normalization \
  --no-ortho-init --log-std-min -20 --log-std-max 20

run test-hopper-iso-obsnorm --env hopper --num-agents 8 --iterations 2000 \
  --no-line-search --no-reward-normalization --no-advantage-normalization \
  --no-ortho-init --log-std-min -20 --log-std-max 20

run test-hopper-iso-rewnorm --env hopper --num-agents 8 --iterations 2000 \
  --no-line-search --no-obs-normalization --no-advantage-normalization \
  --no-ortho-init --log-std-min -20 --log-std-max 20

run test-hopper-iso-advnorm --env hopper --num-agents 8 --iterations 2000 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-ortho-init --log-std-min -20 --log-std-max 20

run test-hopper-iso-orthoinit --env hopper --num-agents 8 --iterations 2000 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --log-std-min -20 --log-std-max 20

run test-hopper-iso-logstdclamp --env hopper --num-agents 8 --iterations 2000 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --no-ortho-init

# --- Crosscheck: full bare on the other two environments ----------------------

run test-swimmer-bare-N1 --env swimmer --num-agents 1 --iterations 1000 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --no-ortho-init --log-std-min -20 --log-std-max 20

run test-swimmer-bare-N8 --env swimmer --num-agents 8 --iterations 1000 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --no-ortho-init --log-std-min -20 --log-std-max 20

run test-humanoid-bare-N1 --env humanoid --num-agents 1 --iterations 1500 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --no-ortho-init --log-std-min -20 --log-std-max 20

run test-humanoid-bare-N8 --env humanoid --num-agents 8 --iterations 1500 \
  --no-line-search --no-obs-normalization --no-reward-normalization \
  --no-advantage-normalization --no-ortho-init --log-std-min -20 --log-std-max 20

# --- Candidate uniform config: bare + observation normalization ---------------

run test-swimmer-iso-obsnorm --env swimmer --num-agents 8 --iterations 1000 \
  --no-line-search --no-reward-normalization --no-advantage-normalization \
  --no-ortho-init --log-std-min -20 --log-std-max 20

run test-humanoid-iso-obsnorm --env humanoid --num-agents 8 --iterations 1500 \
  --no-line-search --no-reward-normalization --no-advantage-normalization \
  --no-ortho-init --log-std-min -20 --log-std-max 20

# -----------------------------------------------------------------------------

if [ "${DRY_RUN:-0}" = 1 ]; then
  for i in "${!NAMES[@]}"; do printf '%s\n\n' "${CMDS[$i]}"; done
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION" >&2
  echo "(or kill it first: tmux kill-session -t $SESSION)" >&2
  exit 1
fi

# Created detached with an explicit size: a detached session defaults to 80x24,
# which runs out of room for new panes after a handful of splits.
tmux new-session -d -s "$SESSION" -x 240 -y 70
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format " #{@name} "

for i in "${!NAMES[@]}"; do
  if [ "$i" -gt 0 ]; then
    tmux split-window -t "$SESSION"
    tmux select-layout -t "$SESSION" tiled   # rebalance so the next split has room
  fi
  # Pane label via a user option, not the pane title: the shell inside can
  # rewrite the title through escape sequences, but not this.
  tmux set-option -p -t "$SESSION" @name "${NAMES[$i]}"
  tmux send-keys -t "$SESSION" "${CMDS[$i]}" C-m
done
tmux select-layout -t "$SESSION" tiled

echo "started ${#NAMES[@]} runs in tmux session '$SESSION'"
echo "  watch:  tmux attach -t $SESSION      (Ctrl-b z zooms a pane, Ctrl-b d detaches)"
echo "  table:  ./sweep_status.py test-"
