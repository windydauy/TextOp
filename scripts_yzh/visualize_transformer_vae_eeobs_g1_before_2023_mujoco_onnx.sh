#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-text_tracker_2}"

POLICY_PATH="${POLICY_PATH:-/home/humanoid/yzh/TextOp/TextOpTracker/logs/rsl_rl/transformer_vae_eeobs_g1_before_2023/2026-05-18_19-55-31_transformer_vae_eeobs_g1_before_2023/latest.onnx}"
MOTION_FOLDER="${MOTION_FOLDER:-/home/humanoid/yzh/TextOp/g1_before_2023}"

TASK="${TASK:-Tracking-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-NMMLP-v0}"
FUTURE_STEPS="${FUTURE_STEPS:-10}"
ANCHOR_BODY_NAME="${ANCHOR_BODY_NAME:-pelvis}"
LIMIT="${LIMIT:-2000}"
START_INDEX="${START_INDEX:-0}"
LOOP="${LOOP:-0}"
REALTIME_FACTOR="${REALTIME_FACTOR:-1.0}"
PAUSE_BETWEEN_MOTIONS="${PAUSE_BETWEEN_MOTIONS:-0.3}"

TRANSFORMER_VAE_PROJECT_ROOT="${TRANSFORMER_VAE_PROJECT_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/transformer_vae}"
TRANSFORMER_VAE_RUN_ROOT="${TRANSFORMER_VAE_RUN_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/outputs/transformer_vae/2026-05-12_16-34-08_optitrack_npz_soma_before_2023}"
TRANSFORMER_VAE_CONFIG_PATH="${TRANSFORMER_VAE_CONFIG_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/params/config.yaml}"
TRANSFORMER_VAE_STATS_PATH="${TRANSFORMER_VAE_STATS_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/stats.npz}"
TRANSFORMER_VAE_ONNX_PATH="${TRANSFORMER_VAE_ONNX_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TRANSFORMER_VAE_BATCH_SIZE="${TRANSFORMER_VAE_BATCH_SIZE:-4096}"

has_motion_files() {
  local root="$1"
  if [[ -f "${root}" && "$(basename "${root}")" == "motion.npz" ]]; then
    return 0
  fi
  [[ -d "${root}" ]] && find "${root}" -name motion.npz -print -quit | grep -q .
}

if ! has_motion_files "${MOTION_FOLDER}"; then
  echo "No motion.npz files found under ${MOTION_FOLDER}" >&2
  exit 1
fi

for required_path in \
  "${POLICY_PATH}" \
  "${MOTION_FOLDER}" \
  "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  "${TRANSFORMER_VAE_CONFIG_PATH}" \
  "${TRANSFORMER_VAE_STATS_PATH}" \
  "${TRANSFORMER_VAE_ONNX_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required visualize path not found: ${required_path}" >&2
    exit 1
  fi
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "/home/humanoid/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/home/humanoid/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

loop_arg=()
if [[ "${LOOP}" == "1" ]]; then
  loop_arg=(--loop)
fi

cd "${ROOT_DIR}/TextOpTracker"
python scripts/visualize_mujoco_onnx_folder.py \
  --policy_path "${POLICY_PATH}" \
  --motion_folder "${MOTION_FOLDER}" \
  --task "${TASK}" \
  --future_steps "${FUTURE_STEPS}" \
  --anchor_body_name "${ANCHOR_BODY_NAME}" \
  --limit "${LIMIT}" \
  --start_index "${START_INDEX}" \
  --realtime_factor "${REALTIME_FACTOR}" \
  --pause_between_motions "${PAUSE_BETWEEN_MOTIONS}" \
  "${loop_arg[@]}" \
  --motion_transformer_vae_project_root "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  --motion_transformer_vae_config_path "${TRANSFORMER_VAE_CONFIG_PATH}" \
  --motion_transformer_vae_stats_path "${TRANSFORMER_VAE_STATS_PATH}" \
  --motion_transformer_vae_encoder_onnx_path "${TRANSFORMER_VAE_ONNX_PATH}" \
  --motion_transformer_vae_batch_size "${TRANSFORMER_VAE_BATCH_SIZE}"
