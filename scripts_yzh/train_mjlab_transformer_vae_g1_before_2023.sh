#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-mjlab-tracker}"
MOTION_ROOT="${MOTION_ROOT:-/home/humanoid/yzh/TextOp/g1_before_2023}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-mjlab_transformer_vae_g1_before_2023_no_motion_freeze_no_wrist_dr}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mjlab_transformer_vae_g1_before_2023}"
WANDB_PROJECT="${WANDB_PROJECT:-MjlabTracker}"
TASK_NAME="${TASK_NAME:-Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-TransformerVAE}"

TRANSFORMER_VAE_PROJECT_ROOT="${TRANSFORMER_VAE_PROJECT_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/transformer_vae}"
TRANSFORMER_VAE_RUN_ROOT="${TRANSFORMER_VAE_RUN_ROOT:-/home/humanoid/yzh/TextOp/motion_ae/outputs/transformer_vae/2026-05-12_16-34-08_optitrack_npz_soma_before_2023}"
TRANSFORMER_VAE_CONFIG_PATH="${TRANSFORMER_VAE_CONFIG_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/params/config.yaml}"
TRANSFORMER_VAE_CKPT_PATH="${TRANSFORMER_VAE_CKPT_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/checkpoints/best_model.pt}"
TRANSFORMER_VAE_STATS_PATH="${TRANSFORMER_VAE_STATS_PATH:-${TRANSFORMER_VAE_RUN_ROOT}/artifacts/stats.npz}"
TRANSFORMER_VAE_BATCH_SIZE="${TRANSFORMER_VAE_BATCH_SIZE:-4096}"

has_motion_files() {
  local root="$1"
  if [[ -f "${root}" ]]; then
    return 0
  fi
  [[ -d "${root}" ]] && find "${root}" -name motion.npz -print -quit | grep -q .
}

if ! has_motion_files "${MOTION_ROOT}"; then
  echo "No motion.npz files found under ${MOTION_ROOT}" >&2
  exit 1
fi

for required_path in \
  "${ROOT_DIR}/MjlabTracker" \
  "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  "${TRANSFORMER_VAE_CONFIG_PATH}" \
  "${TRANSFORMER_VAE_CKPT_PATH}" \
  "${TRANSFORMER_VAE_STATS_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path not found: ${required_path}" >&2
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

export PYTHONPATH="${ROOT_DIR}/MjlabTracker/source/latent_tracker:/home/humanoid/yzh/mjlab/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-codex}"
export WANDB_USERNAME="${WANDB_USERNAME:-yzh_academic-shanghai-jiao-tong-university}"
export WANDB_ENTITY="${WANDB_ENTITY:-yzh_academic-shanghai-jiao-tong-university}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${ROOT_DIR}/scripts/train.sh" ]]; then
  EXISTING_WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "${ROOT_DIR}/scripts/train.sh" | head -n 1)"
  if [[ -n "${EXISTING_WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY="${EXISTING_WANDB_API_KEY}"
  fi
fi

cd "${ROOT_DIR}/MjlabTracker"

GPU_IDS_ARG="${GPU_IDS:-}"
if [[ -z "${GPU_IDS_ARG}" ]]; then
  case "${DEVICE}" in
    cpu)
      GPU_IDS_ARG="None"
      ;;
    cuda:*)
      GPU_IDS_ARG="[${DEVICE#cuda:}]"
      ;;
    *)
      GPU_IDS_ARG="[0]"
      ;;
  esac
fi

HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py "${TASK_NAME}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --env.commands.motion.motion-files "[\"${MOTION_ROOT}\"]" \
  --env.commands.motion.anchor-body-name "torso_link" \
  --env.commands.motion.future-steps 10 \
  --env.commands.motion.motion-ae-enabled False \
  --env.commands.motion.motion-transformer-vae-enabled True \
  --env.commands.motion.motion-transformer-vae-project-root "${TRANSFORMER_VAE_PROJECT_ROOT}" \
  --env.commands.motion.motion-transformer-vae-config-path "${TRANSFORMER_VAE_CONFIG_PATH}" \
  --env.commands.motion.motion-transformer-vae-checkpoint-path "${TRANSFORMER_VAE_CKPT_PATH}" \
  --env.commands.motion.motion-transformer-vae-stats-path "${TRANSFORMER_VAE_STATS_PATH}" \
  --env.commands.motion.motion-transformer-vae-latent-mode z_c \
  --env.commands.motion.motion-transformer-vae-batch-size "${TRANSFORMER_VAE_BATCH_SIZE}" \
  --env.commands.motion.random-static-prob -1.0 \
  --env.commands.motion.enable-adaptive-sampling True \
  --env.commands.motion.ads-type v2 \
  --env.commands.motion.adaptive-beta 0.5 \
  --env.commands.motion.adaptive-alpha 0.1 \
  --env.commands.motion.adaptive-uniform-ratio 0.1 \
  --env.commands.motion.freeze-frame-aug False \
  --env.commands.motion.freeze-frame-aug-prob 0.0 \
  --env.events.randomize-rigid-body-mass.params.ranges "(1.0,1.0)" \
  --agent.experiment-name "${EXPERIMENT_NAME}" \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  --agent.run-name "${RUN_NAME}" \
  --agent.logger wandb \
  --agent.wandb-project "${WANDB_PROJECT}" \
  --agent.actor.hidden-dims "(4096,2048,1024,512,256)" \
  --agent.critic.hidden-dims "(4096,2048,1024,512,256)" \
  --agent.seed "${SEED:-42}" \
  --gpu-ids "${GPU_IDS_ARG}" \
  "$@"
