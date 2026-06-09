#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-text_tracker_2}"

POLICY_PATH="${POLICY_PATH:-/home/humanoid/yzh/TextOp/TextOpTracker/logs/rsl_rl/motion_transformer_vae_zc_trip_npz_filtered/2026-05-11_12-59-33_motion_transformer_vae_zc_trip_npz_filtered/latest.onnx}"
MOTION_ROOT="${MOTION_ROOT:-/home/humanoid/yzh/TextOp/optritrack_dataset/optritrack_npz_filtered}"
MOTION_PATH="${MOTION_PATH:-/home/humanoid/yzh/TextOp/trip_npz_filtered/converted_0110/balance_001_Skeleton 006_z_up_x_forward_gym_1}"

TRANSFORMER_VAE_PROJECT_ROOT="${TRANSFORMER_VAE_PROJECT_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/transformer_vae}"
TRANSFORMER_VAE_RUN_ROOT="${TRANSFORMER_VAE_RUN_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/outputs/transformer_vae/2026-05-11_00-19-00_optitrack_npz_trip_filtered}"
TRANSFORMER_VAE_CONFIG_PATH="${TRANSFORMER_VAE_CONFIG_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/params/config.yaml}"
TRANSFORMER_VAE_CKPT_PATH="${TRANSFORMER_VAE_CKPT_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/checkpoints/best_model.pt}"
TRANSFORMER_VAE_STATS_PATH="${TRANSFORMER_VAE_STATS_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/stats.npz}"
TRANSFORMER_VAE_ONNX_PATH="${TRANSFORMER_VAE_ONNX_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TRANSFORMER_VAE_BATCH_SIZE="${TRANSFORMER_VAE_BATCH_SIZE:-4096}"
FUTURE_STEPS="${FUTURE_STEPS:-10}"
ANCHOR_BODY_NAME="${ANCHOR_BODY_NAME:-pelvis}"
FORCE_EXPORT_VAE_ONNX="${FORCE_EXPORT_VAE_ONNX:-0}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "/home/humanoid/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/home/humanoid/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

if [[ -z "${MOTION_PATH}" ]]; then
  MOTION_PATH="$(find "${MOTION_ROOT}" -name motion.npz -print -quit)"
fi

for required_path in \
  "${POLICY_PATH}" \
  "${MOTION_PATH}" \
  "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  "${TRANSFORMER_VAE_CONFIG_PATH}" \
  "${TRANSFORMER_VAE_CKPT_PATH}" \
  "${TRANSFORMER_VAE_STATS_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required deploy path not found: ${required_path}" >&2
    exit 1
  fi
done

if [[ "${FORCE_EXPORT_VAE_ONNX}" == "1" || ! -f "${TRANSFORMER_VAE_ONNX_PATH}" ]]; then
  python "${ROOT_DIR}/scripts/export_motion_transformer_vae_onnx.py" \
    --project_root "${TRANSFORMER_VAE_PROJECT_ROOT}" \
    --config_path "${TRANSFORMER_VAE_CONFIG_PATH}" \
    --checkpoint_path "${TRANSFORMER_VAE_CKPT_PATH}" \
    --stats_path "${TRANSFORMER_VAE_STATS_PATH}" \
    --output_path "${TRANSFORMER_VAE_ONNX_PATH}" \
    --latent_mode z_c
fi

cd "${ROOT_DIR}/TextOpTracker"
python scripts/deploy_mujoco.py \
  --policy_path "${POLICY_PATH}" \
  --motion_path "${MOTION_PATH}" \
  --task Tracking-Flat-G1-ProjGravAnchorObs-TransformerVAE-NMMLP-v0 \
  --future_steps "${FUTURE_STEPS}" \
  --anchor_body_name "${ANCHOR_BODY_NAME}" \
  --motion_transformer_vae_project_root "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  --motion_transformer_vae_config_path "${TRANSFORMER_VAE_CONFIG_PATH}" \
  --motion_transformer_vae_stats_path "${TRANSFORMER_VAE_STATS_PATH}" \
  --motion_transformer_vae_encoder_onnx_path "${TRANSFORMER_VAE_ONNX_PATH}" \
  --motion_transformer_vae_batch_size "${TRANSFORMER_VAE_BATCH_SIZE}"
