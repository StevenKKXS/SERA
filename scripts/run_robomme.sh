#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
SIM_PYTHON="${SIM_PYTHON:-${REPO_ROOT}/.venv-sim/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-${REPO_ROOT}/.venv-policy/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/robomme}"
TASKS="${TASKS:-MoveCube}"
NUM_EPISODES="${NUM_EPISODES:-1}"
PLANNER_PORT="${PLANNER_PORT:-18229}"
POLICY_PORT="${POLICY_PORT:-19300}"
PLANNER_TOKEN_FILE="${PLANNER_TOKEN_FILE:-${RUN_ROOT}/planner.token}"

case "${1:-help}" in
  capture)
    export CUDA_VISIBLE_DEVICES="${SIM_GPU:-0}"
    exec "${SIM_PYTHON}" scripts/codex_init_video/capture_init.py \
      --tasks "${TASKS}" --num-episodes "${NUM_EPISODES}" \
      --output-dir "${RUN_ROOT}/init_inputs"
    ;;
  init)
    : "${PLANNER_MODEL:?Set PLANNER_MODEL to a model available to your planner CLI}"
    exec "${SIM_PYTHON}" scripts/codex_init_video/run_init_video.py run \
      --input-jsonl "${RUN_ROOT}/init_inputs/manifest.jsonl" \
      --output-dir "${RUN_ROOT}/init_results" --cache-dir "${RUN_ROOT}/init_cache" \
      --prompt-profile planner_guidance_v3 --model "${PLANNER_MODEL}" \
      --reasoning-effort high
    ;;
  planner)
    : "${PLANNER_MODEL:?Set PLANNER_MODEL to a model available to your planner CLI}"
    extra_args=()
    profile=retrieval_v8_5_refine_then_lock
    if [[ -n "${INTERVENTION_BUNDLE:-}" ]]; then
      profile=self_evolve_v4_fullstack_trainval
      extra_args+=(--runtime-intervention-bundle "${INTERVENTION_BUNDLE}")
    fi
    exec "${SIM_PYTHON}" scripts/codex_rollout/planner_server.py \
      --host 127.0.0.1 --port "${PLANNER_PORT}" --token-file "${PLANNER_TOKEN_FILE}" \
      --planner-kind text-progress --text-progress-profile "${profile}" \
      --retrieval-scheme R0 --retrieval-top-k 5 --model "${PLANNER_MODEL}" \
      --reasoning-effort high --timeout-seconds 900 "${extra_args[@]}"
    ;;
  policy)
    : "${CHECKPOINT:?Set CHECKPOINT to the extracted symbolic-grounded-subgoal/79999 directory}"
    export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
    exec "${POLICY_PYTHON}" scripts/serve_policy.py --seed=7 --port="${POLICY_PORT}" \
      policy:checkpoint --policy.dir="${CHECKPOINT}" --policy.config=mme_vla_suite
    ;;
  eval)
    export CUDA_VISIBLE_DEVICES="${SIM_GPU:-0}"
    exec "${SIM_PYTHON}" scripts/codex_rollout/run_robomme_batch.py \
      --repo-root "${REPO_ROOT}" --benchmark-root "${REPO_ROOT}/third_party/robomme_benchmark" \
      --init-results-dir "${RUN_ROOT}/init_results" --run-dir "${RUN_ROOT}/evaluation" \
      --planner-url "http://127.0.0.1:${PLANNER_PORT}" --planner-token-file "${PLANNER_TOKEN_FILE}" \
      --pi05-host 127.0.0.1 --pi05-port "${POLICY_PORT}" \
      --tasks "${TASKS}" --num-episodes "${NUM_EPISODES}" \
      --planner-memory-mode text_progress --obs-horizon 16 --planner-frame-stride 2 \
      --max-steps 1300 --video-episodes 0
    ;;
  summary)
    exec "${SIM_PYTHON}" scripts/codex_rollout/summarize_results.py \
      --run-dir "${RUN_ROOT}/evaluation" --tasks "${TASKS}" --num-episodes "${NUM_EPISODES}" \
      --planner-memory-mode text_progress
    ;;
  help|--help|-h)
    printf '%s\n' 'Usage: bash scripts/run_robomme.sh {capture|init|planner|policy|eval|summary}' \
      'Configuration: TASKS, NUM_EPISODES, RUN_ROOT, PLANNER_MODEL, CHECKPOINT,' \
      'SIM_PYTHON, POLICY_PYTHON, POLICY_GPU, SIM_GPU, PLANNER_PORT, POLICY_PORT,' \
      'INTERVENTION_BUNDLE (optional selected harness).'
    ;;
  *) printf 'Unknown command: %s\n' "$1" >&2; exit 2 ;;
esac
