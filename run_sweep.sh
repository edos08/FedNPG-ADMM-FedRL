#!/usr/bin/env bash
# Sweep runner: one invocation per tmux session.
#   ./run_sweep.sh swimmer|hopper|humanoid   -> FedNPG-ADMM, all agent counts
#   ./run_sweep.sh npg                       -> plain-TRPO baselines, all envs
#
# Resumable: a run whose metadata.json already has completed_at is skipped,
# so re-running after a crash or a reboot picks up where it left off.
set -u
TARGET="${1:?usage: run_sweep.sh swimmer|hopper|humanoid|npg}"
SEEDS="${SEEDS:-0 1 2}"
AGENTS="${AGENTS:-1 2 4 8}"
DEVICE="${DEVICE:-cuda:0}"
EVAL="--eval-interval 25 --eval-episodes 10 --save-checkpoint"

# Four trainings run concurrently and every N=8 run forks 8 env subprocesses.
# Without this, each torch process grabs all 46 cores and they thrash each
# other -- MuJoCo stepping is CPU-bound, so oversubscription costs real
# wall-clock.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

ADMM_DIR=/mnt/edoardo/FedNPG-ADMM-FedRL
NPG_DIR=/mnt/edoardo/NPG

done_already() {  # $1 = results dir, $2 = run id
  [ -f "$1/$2/metadata.json" ] && grep -q '"completed_at"' "$1/$2/metadata.json"
}

if [ "$TARGET" = npg ]; then
  cd "$NPG_DIR"
  for s in $SEEDS; do
    for env in swimmer hopper humanoid; do
      # Table 4 sets rho = 0.01 for Humanoid and 0.1 elsewhere. fedrl-train-admm
      # reads that per --env and uses it as the CG damping too; npg_trpo is a
      # separate project whose --damping defaults to SB3's 0.1, so Humanoid has
      # to be matched by hand or the with/without-ADMM comparison silently
      # differs in the damping as well as in the algorithm.
      damping=0.1; [ "$env" = humanoid ] && damping=0.01
      id="${env}-npg-s${s}"
      done_already results "$id" && { echo "[skip] $id"; continue; }
      echo "=== $(date '+%F %T')  start $id  (damping $damping) ==="
      npg-train --env "$env" --seed "$s" --device "$DEVICE" --damping "$damping" \
        $EVAL --run-id "$id"
    done
  done
else
  cd "$ADMM_DIR"
  for s in $SEEDS; do
    for n in $AGENTS; do
      id="${TARGET}-N${n}-s${s}"
      done_already results "$id" && { echo "[skip] $id"; continue; }
      echo "=== $(date '+%F %T')  start $id ==="
      fedrl-train-admm --env "$TARGET" --num-agents "$n" --seed "$s" \
        --device "$DEVICE" $EVAL --run-id "$id"
    done
  done
fi
echo "=== $(date '+%F %T')  $TARGET sweep finished ==="
