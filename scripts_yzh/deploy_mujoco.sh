cd ./TextOpTracker
# python scripts/deploy_mujoco.py \
#   --policy_path "/home/humanoid/yzh/TextOp/TextOpTracker/logs/rsl_rl/Run_npz_filtered_jv/2026-04-29_22-40-16_Run_npz_filtered_jv/exported/policy.onnx" \
#   --motion_path "/home/humanoid/yzh/TextOp/motion_ae/optitrack_npz_filtered_all/SQUAT6_Skeleton_z_up_x_forward_gym" \
#   --obs_config "ProjGravObs" \
#   --future_steps 10 \
#   --anchor_body_name "pelvis"
python scripts/deploy_mujoco.py \
  --policy_path "/home/humanoid/yzh/TextOp/TextOpTracker/logs/rsl_rl/direct_tracker/Run_npz_filtered_before_0110_ProjGravObs/2026-05-07_11-06-05_Run_npz_filtered_before_0110_ProjGravObs/latest.onnx" \
  --motion_path "/home/humanoid/yzh/TextOp/kimomotion_dataset/zero_shot_mujoco_npz/g1_generated_proposal/" \
  --task Tracking-Flat-G1-ProjGravObs-MNMLP-v0 \
  --future_steps 10
# python scripts/deploy_mujoco.py \
#   --policy_path /home/humanoid/yzh/TextOp/TextOpTracker/logs/rsl_rl/motion_ae_zdequant/2026-05-08_23-13-25_motion_ae_zdequant/latest.onnx \
#   --motion_path "/home/humanoid/yzh/TextOp/motion_ae/optitrack_npz_filtered_all/SQUAT6_Skeleton_z_up_x_forward_gym/" \
#   --task Tracking-Flat-G1-ProjGravAnchorObs-MotionAE-NMMLP-v0

# /home/humanoid/yzh/TextOp/TextOp-Data/TextOpTracker/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/exported/policy.onnx
#/home/humanoid/yzh/TextOp/exported/policy.onnx
