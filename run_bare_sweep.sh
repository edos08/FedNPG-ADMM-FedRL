#!/usr/bin/env bash
# Sweep runner for the "paper-literal" configuration -- Algorithm 1 with every
# TRPO/stable-baselines3 practical setting this project added turned back off.
#
#   ./run_bare_sweep.sh hopper      -> bare FedNPG-ADMM, all agent counts, all seeds
#   ./run_bare_sweep.sh isolate     -> add ONE extra back at a time, N=8, seed 0
#   ./run_bare_sweep.sh crosscheck  -> bare on swimmer/humanoid, N=1 and N=8
#
# Why this exists. Under the default (TRPO-practical) configuration Hopper
# reaches 3854 deterministic return at iteration 575 with num_agents=8 and then
# decays 51% to 1870 by iteration 2000, while num_agents=1 holds 3639 flat --
# the paper's agent-count ordering, inverted. With every added practical
# setting OFF the decay disappears entirely (-3%) and the ordering comes back.
# That result is one seed per agent count, and the within-run swings are large
# (num_agents=2 moved between 1218 and 3065 across evaluations of the SAME
# run), so it is not reportable until it is multi-seed. That is what `hopper`
# below is for.
#
# Every run spells its flags out in full rather than sharing a variable: an
# earlier isolation attempt reused a $BARE string that still contained
# --no-line-search, so the "plus line search" run was a byte-identical
# duplicate of the bare one and the experiment silently did not happen.
#
# Resumable: a run whose metadata.json already has completed_at is skipped.
set -u
TARGET="${1:?usage: run_bare_sweep.sh hopper|isolate|crosscheck}"
SEEDS="${SEEDS:-0 1 2 3 4}"
AGENTS="${AGENTS:-1 2 4 8}"
DEVICE="${DEVICE:-cpu}"
# How many trainings to keep in flight. Each num_agents=N run forks N env
# subprocesses, so a wave of 4 at AGENTS="1 2 4 8" is already 15 processes.
WAVE="${WAVE:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd /mnt/edoardo/FedNPG-ADMM-FedRL

# The six settings this project added on top of Algorithm 1, all off. Keep this
# list and the isolation targets below in sync by hand -- they are deliberately
# not derived from a shared variable (see header).
BARE_FLAGS=(
  --no-line-search
  --no-obs-normalization
  --no-reward-normalization
  --no-advantage-normalization
  --no-ortho-init
  --log-std-min -20 --log-std-max 20
  --on-invalid-step skip
)
EVAL_FLAGS=(--eval-interval 25 --eval-episodes 10 --save-checkpoint)

done_already() { [ -f "results/$1/metadata.json" ] && grep -q '"completed_at"' "results/$1/metadata.json"; }

# Block until fewer than $WAVE trainings are running, so the machine is not
# oversubscribed by launching every run at once.
throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$WAVE" ]; do sleep 20; done
}

launch() {  # $1 = run id, rest = flags
  local id="$1"; shift
  if done_already "$id"; then echo "[skip] $id"; return; fi
  throttle
  echo "=== $(date '+%F %T')  start $id"
  fedrl-train-admm "$@" --run-id "$id" > "results/${id}.log" 2>&1 &
}

case "$TARGET" in
  hopper)
    for s in $SEEDS; do
      for n in $AGENTS; do
        launch "hopper-bare-N${n}-s${s}" \
          --env hopper --num-agents "$n" --iterations 2000 --seed "$s" \
          --device "$DEVICE" "${BARE_FLAGS[@]}" "${EVAL_FLAGS[@]}"
      done
    done
    ;;

  isolate)
    # One added setting restored per run, everything else still off, so a
    # difference against hopper-bare-N8-s0 is attributable to exactly one
    # component. num_agents=8 because that is where the decay is largest.
    launch "hopper-iso-linesearch" --env hopper --num-agents 8 --iterations 2000 --seed 0 \
      --device "$DEVICE" --on-invalid-step skip \
      --no-obs-normalization --no-reward-normalization --no-advantage-normalization \
      --no-ortho-init --log-std-min -20 --log-std-max 20 "${EVAL_FLAGS[@]}"

    launch "hopper-iso-obsnorm" --env hopper --num-agents 8 --iterations 2000 --seed 0 \
      --device "$DEVICE" --on-invalid-step skip --no-line-search \
      --no-reward-normalization --no-advantage-normalization \
      --no-ortho-init --log-std-min -20 --log-std-max 20 "${EVAL_FLAGS[@]}"

    launch "hopper-iso-rewnorm" --env hopper --num-agents 8 --iterations 2000 --seed 0 \
      --device "$DEVICE" --on-invalid-step skip --no-line-search \
      --no-obs-normalization --no-advantage-normalization \
      --no-ortho-init --log-std-min -20 --log-std-max 20 "${EVAL_FLAGS[@]}"

    launch "hopper-iso-advnorm" --env hopper --num-agents 8 --iterations 2000 --seed 0 \
      --device "$DEVICE" --on-invalid-step skip --no-line-search \
      --no-obs-normalization --no-reward-normalization \
      --no-ortho-init --log-std-min -20 --log-std-max 20 "${EVAL_FLAGS[@]}"

    launch "hopper-iso-logstdclamp" --env hopper --num-agents 8 --iterations 2000 --seed 0 \
      --device "$DEVICE" --on-invalid-step skip --no-line-search \
      --no-obs-normalization --no-reward-normalization --no-advantage-normalization \
      --no-ortho-init "${EVAL_FLAGS[@]}"
    ;;

  crosscheck)
    # Does the bare configuration break the two environments that currently
    # reproduce the paper? Observation normalization was added because Swimmer
    # plateaued far below Table 2 without it, so this is a real risk, and the
    # answer decides whether the report can use ONE configuration throughout
    # or has to use a different one for Hopper and say so.
    for env in swimmer humanoid; do
      for n in 1 8; do
        launch "${env}-bare-N${n}-s0" \
          --env "$env" --num-agents "$n" --seed 0 \
          --device "$DEVICE" "${BARE_FLAGS[@]}" "${EVAL_FLAGS[@]}"
      done
    done
    ;;

  *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac

wait
echo "=== $(date '+%F %T')  all done"
