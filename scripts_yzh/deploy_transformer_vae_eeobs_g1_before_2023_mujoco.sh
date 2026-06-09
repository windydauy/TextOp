#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-text_tracker_2}"

POLICY_PATH="${POLICY_PATH:-/home/humanoid/yzh/TextOp/TextOpTracker/logs/rsl_rl/transformer_vae_eeobs_g1_before_2023/2026-05-18_19-55-31_transformer_vae_eeobs_g1_before_2023/latest.onnx}"
MOTION_PATH="${MOTION_PATH:-/home/humanoid/yzh/TextOp/kimomotion_dataset/zero_shot_mujoco_npz/g1_generated_proposal/}"

TASK="${TASK:-Tracking-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-NMMLP-v0}"
FUTURE_STEPS="${FUTURE_STEPS:-10}"
ANCHOR_BODY_NAME="${ANCHOR_BODY_NAME:-pelvis}"

TRANSFORMER_VAE_PROJECT_ROOT="${TRANSFORMER_VAE_PROJECT_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/transformer_vae}"
TRANSFORMER_VAE_RUN_ROOT="${TRANSFORMER_VAE_RUN_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/outputs/transformer_vae/2026-05-12_16-34-08_optitrack_npz_soma_before_2023}"
TRANSFORMER_VAE_CONFIG_PATH="${TRANSFORMER_VAE_CONFIG_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/params/config.yaml}"
TRANSFORMER_VAE_CKPT_PATH="${TRANSFORMER_VAE_CKPT_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/checkpoints/best_model.pt}"
TRANSFORMER_VAE_STATS_PATH="${TRANSFORMER_VAE_STATS_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/stats.npz}"
TRANSFORMER_VAE_ONNX_PATH="${TRANSFORMER_VAE_ONNX_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TRANSFORMER_VAE_BATCH_SIZE="${TRANSFORMER_VAE_BATCH_SIZE:-4096}"

for required_path in \
  "${POLICY_PATH}" \
  "${MOTION_PATH}" \
  "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  "${TRANSFORMER_VAE_CONFIG_PATH}" \
  "${TRANSFORMER_VAE_CKPT_PATH}" \
  "${TRANSFORMER_VAE_STATS_PATH}" \
  "${TRANSFORMER_VAE_ONNX_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required deploy path not found: ${required_path}" >&2
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

cd "${ROOT_DIR}/TextOpTracker"
python scripts/deploy_mujoco.py \
  --policy_path "${POLICY_PATH}" \
  --motion_path "${MOTION_PATH}" \
  --task "${TASK}" \
  --future_steps "${FUTURE_STEPS}" \
  --anchor_body_name "${ANCHOR_BODY_NAME}" \
  --motion_transformer_vae_project_root "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  --motion_transformer_vae_config_path "${TRANSFORMER_VAE_CONFIG_PATH}" \
  --motion_transformer_vae_stats_path "${TRANSFORMER_VAE_STATS_PATH}" \
  --motion_transformer_vae_encoder_onnx_path "${TRANSFORMER_VAE_ONNX_PATH}" \
  --motion_transformer_vae_batch_size "${TRANSFORMER_VAE_BATCH_SIZE}"
