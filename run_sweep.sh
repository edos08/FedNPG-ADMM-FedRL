#!/usr/bin/env bash
# Swimmer and Humanoid sweeps, in the configuration the published results use.
#
#   ./run_sweep.sh swimmer|humanoid          FedNPG-ADMM, every agent count
#   ./run_sweep.sh base-swimmer|base-humanoid  standard-average FedNPG + FedPPO, N=8
#
# Hopper uses a reduced configuration and has its own runner: final_sweep.sh.
#
# Environment overrides: SEEDS (default "0 1 2 3 4"), AGENTS (default
# "1 2 4 8"), DEVICE (default cpu). Runs go to ./results/<run-id>/.
#
# Resumable: a run whose metadata.json already has completed_at is skipped,
# so re-running after a crash or a reboot picks up where it left off.
set -u
TARGET="${1:?usage: run_sweep.sh swimmer|humanoid|base-swimmer|base-humanoid}"
SEEDS="${SEEDS:-0 1 2 3 4}"
AGENTS="${AGENTS:-1 2 4 8}"
# CPU by default. The policies are small and MuJoCo stepping is CPU-bound, so
# a GPU mostly adds per-step launch overhead; on our machine CPU was faster
# on all three environments at N=8.
DEVICE="${DEVICE:-cpu}"
EVAL="--eval-interval 25 --eval-episodes 10 --save-checkpoint"

# One thread per training: the workload is not thread-parallel, and several
# concurrent runs (each forking N environment processes) otherwise
# oversubscribe the CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")"

done_already() {  # $1 = run id
  [ -f "results/$1/metadata.json" ] && grep -q '"completed_at"' "results/$1/metadata.json"
}

case "$TARGET" in
  hopper|base-hopper)
    echo "Hopper uses its own configuration: run ./final_sweep.sh hopper" \
         "(or hopper-baselines) instead." >&2
    exit 2
    ;;
  base-swimmer|base-humanoid)
    # The paper compares the three methods at N=8.
    env="${TARGET#base-}"
    for s in $SEEDS; do
      for algo in fednpg-standard fedppo; do
        id="${env}-${algo}-N8-s${s}"
        done_already "$id" && { echo "[skip] $id"; continue; }
        echo "=== $(date '+%F %T')  start $id ==="
        fedrl-train-baseline --algo "$algo" --env "$env" --num-agents 8 --seed "$s" \
          --device "$DEVICE" $EVAL --run-id "$id"
      done
    done
    ;;
  swimmer|humanoid)
    for s in $SEEDS; do
      for n in $AGENTS; do
        id="${TARGET}-N${n}-s${s}"
        done_already "$id" && { echo "[skip] $id"; continue; }
        echo "=== $(date '+%F %T')  start $id ==="
        fedrl-train-admm --env "$TARGET" --num-agents "$n" --seed "$s" \
          --device "$DEVICE" $EVAL --run-id "$id"
      done
    done
    ;;
  *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac
echo "=== $(date '+%F %T')  $TARGET sweep finished ==="
