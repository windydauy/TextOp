#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-text_tracker_2}"
MOTION_ROOT="${MOTION_ROOT:-/home/humanoid/yzh/TextOp/g1_before_2023}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-direct_ref_motion_eeobs_g1_before_2023}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-direct_ref_motion_eeobs_g1_before_2023}"

has_motion_files() {
  local root="$1"
  if [[ -f "${root}" && "$(basename "${root}")" == "motion.npz" ]]; then
    return 0
  fi
  [[ -d "${root}" ]] && find "${root}" -name motion.npz -print -quit | grep -q .
}

if ! has_motion_files "${MOTION_ROOT}"; then
  echo "No motion.npz files found under ${MOTION_ROOT}" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "/home/humanoid/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/home/humanoid/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

export WANDB_USERNAME="${WANDB_USERNAME:-yzh_academic-shanghai-jiao-tong-university}"
export WANDB_ENTITY="${WANDB_ENTITY:-yzh_academic-shanghai-jiao-tong-university}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${ROOT_DIR}/scripts/train.sh" ]]; then
  EXISTING_WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "${ROOT_DIR}/scripts/train.sh" | head -n 1)"
  if [[ -n "${EXISTING_WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY="${EXISTING_WANDB_API_KEY}"
  fi
fi

cd "${ROOT_DIR}/TextOpTracker"

HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --headless \
  --logger wandb \
  --log_project_name TextOpTracker \
  --task=Tracking-Flat-G1-ProjGravAnchorEEObs-DirectRefMotion-NMMLP-v0 \
  --motion_file="${MOTION_ROOT}" \
  --run_name "${RUN_NAME}" \
  agent.experiment_name="${EXPERIMENT_NAME}" \
  agent.max_iterations="${MAX_ITERATIONS}" \
  --num_envs="${NUM_ENVS}" \
  env.commands.motion.anchor_body_name="pelvis" \
  env.commands.motion.future_steps=10 \
  env.commands.motion.motion_ae_enabled=False \
  env.commands.motion.motion_transformer_vae_enabled=False \
  env.commands.motion.random_static_prob=-1.0 \
  env.commands.motion.enable_adaptive_sampling=True \
  env.commands.motion.ads_type=v2 \
  env.commands.motion.adaptive_beta=0.5 \
  env.commands.motion.adaptive_alpha=0.1 \
  env.commands.motion.adaptive_uniform_ratio=0.1 \
  agent.policy.actor_hidden_dims=[4096,2048,1024,512,256] \
  agent.policy.critic_hidden_dims=[4096,2048,1024,512,256] \
  --seed=42 \
  --device="${DEVICE}"
