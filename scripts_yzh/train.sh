# export WANDB_MODE=offline
export WANDB_USERNAME=yzh_academic-shanghai-jiao-tong-university
export WANDB_ENTITY=yzh_academic-shanghai-jiao-tong-university  # 可留着，但当前这份代码主要读上面那个

cd ./TextOpTracker

HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --headless \
  --logger ternsorboard \
  --log_project_name TextOpTracker \
  --task=Tracking-Flat-G1-ProjGravAnchorObs-NMMLP-v0 \
  --motion_file='/home/humanoid/yzh/TextOp/optritrack_dataset/optitrack_csv_filtered/' \
  --run_name Run_npz_filtered_jv \
  agent.experiment_name=Run_npz_filtered_jv \
  agent.max_iterations=100000 \
  --num_envs=8192 \
  env.commands.motion.anchor_body_name="pelvis" \
  env.commands.motion.future_steps=10 \
  env.commands.motion.random_static_prob=-1.0 \
  env.rewards.feet_slide.params.pfail_threshold=1.0 \
  env.rewards.soft_landing.params.pfail_threshold=1.0 \
  env.rewards.overspeed.params.pfail_threshold=1.0 \
  env.rewards.overeffort.params.pfail_threshold=1.0 \
  env.rewards.feet_slide.weight=-0.3 \
  env.rewards.soft_landing.weight=-0.0003 \
  env.rewards.overspeed.weight=-1.0 \
  env.rewards.overeffort.weight=-1.0 \
  env.commands.motion.enable_adaptive_sampling=True \
  env.commands.motion.ads_type=v2 \
  env.commands.motion.adaptive_beta=0.5 \
  env.commands.motion.adaptive_alpha=0.1 \
  env.commands.motion.adaptive_uniform_ratio=0.1 \
  agent.policy.actor_hidden_dims=[4096,2048,1024,512,256] \
  agent.policy.critic_hidden_dims=[4096,2048,1024,512,256] \
  --seed=42 \
