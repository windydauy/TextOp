# TextOp 与 Sonic Adaptive Sampling 对比

本文总结 TextOp 当前 adaptive sampling 的参数含义、与 Gear Sonic release 版本的差异，以及在大数据量训练中建议增加的采样概率上限。

## TextOp 当前方法

TextOp 的 adaptive sampling 在 `commands_multi.py` 中实现。当前训练脚本使用的是：

```text
ads_type = v2
adaptive_alpha = 0.1
adaptive_beta = 0.5
adaptive_uniform_ratio = 0.1
```

`ads_type=v2` 的核心公式是：

```text
failed_count  = alpha * current_failed  + (1 - alpha) * old_failed
success_count = alpha * current_success + (1 - alpha) * old_success

p_fail = failed_count / (failed_count + success_count)

p_adaptive = p_fail ** beta
p_adaptive = p_adaptive / sum(p_adaptive)

p_sample = (1 - uniform_ratio) * p_adaptive
         + uniform_ratio * uniform
```

参数含义：

- `adaptive_alpha` 控制失败/成功统计的更新速度。`0.1` 更新很快，少数最近失败的 clips 会迅速获得高采样概率。
- `adaptive_beta` 控制失败率到采样概率的平滑程度。`beta < 1` 会压平差异，`beta > 1` 会更集中到高失败率 clips。
- `adaptive_uniform_ratio` 控制均匀采样保底比例。`0.1` 表示 90% 概率来自失败率分布，10% 来自均匀分布。

在 9933 个 motions 的训练中，`adaptive_alpha=0.1` 偏激进。实际 run 中单个 motion 的采样概率已经从接近均匀迅速集中：

```text
model_500  top1 prob ~= 6.6x uniform
model_1000 top1 prob ~= 36x uniform
model_1500 top1 prob ~= 117x uniform
model_2000 top1 prob ~= 307x uniform
```

这会把训练分布推向少数高失败 motion，例如 `jump_on_50cm`、`jump_off_50cm`、`injured leg`、`one_leg_jumping`、高难 dance/jump clips。对应现象是 `ee_body_pos` termination 上升，`mean_episode_length` 和 reward 在 1000 step 后下降。

## Sonic Release 方法

Sonic release 配置：

```yaml
motion_lib_cfg:
  adaptive_sampling:
    adp_samp_failure_rate_max_over_mean: 200
```

它继承的默认 adaptive sampling 还包括：

```yaml
adaptive_sampling:
  enable: true
  bin_size: 50
  sequence_length_agnostic: true
  init_num_failures: 1
  uniform_sampling_rate: 0.1
  adp_samp_failure_rate_max_over_mean: 50.0
```

Sonic 的关键差异：

- Sonic 是 frame/bin 级别采样，每个 motion 会按 50 frames 切分；TextOp 当前是 whole-motion 级别。
- Sonic 使用累计统计：`failure_rate = num_failures / num_episodes`；TextOp 当前使用 EMA。
- Sonic 有 failure-rate 上限：`failure_rate <= mean_failure_rate * adp_samp_failure_rate_max_over_mean`。
- Sonic 同样混入 10% uniform sampling。

因此 Sonic release 的 `200` 仍然偏激进，但它有累计统计和 failure-rate cap；TextOp 当前 `alpha=0.1` 且没有 hard cap，实际更容易快速集中到少数 clips。

## 建议的 TextOp 修正

第一阶段建议先做稳定性 ablation：

```bash
env.commands.motion.enable_adaptive_sampling=False
env.commands.motion.freeze_frame_aug=False
env.commands.motion.freeze_frame_aug_prob=0.0
env.events.randomize_rigid_body_mass.params.asset_cfg.body_names="torso_link"
```

如果确认关闭 adaptive 后不再塌，再用更温和的 adaptive：

```bash
env.commands.motion.enable_adaptive_sampling=True
env.commands.motion.ads_type=v2
env.commands.motion.adaptive_alpha=0.001
env.commands.motion.adaptive_beta=0.5
env.commands.motion.adaptive_uniform_ratio=0.3
env.commands.motion.max_prob_over_uniform=50
```

更保守版本：

```bash
env.commands.motion.adaptive_alpha=0.001
env.commands.motion.adaptive_beta=0.25
env.commands.motion.adaptive_uniform_ratio=0.5
env.commands.motion.max_prob_over_uniform=20
```

## `max_prob_over_uniform`

新增参数：

```text
env.commands.motion.max_prob_over_uniform
```

含义是限制单个 motion 的最大采样概率不能超过均匀采样概率的 N 倍：

```text
p_i <= max_prob_over_uniform / num_motion
```

例如有 9933 个 motions：

```text
uniform prob ~= 0.0001007
max_prob_over_uniform=50  => 单 motion 最大概率 ~= 0.0050
max_prob_over_uniform=20  => 单 motion 最大概率 ~= 0.0020
```

这个参数默认是 `0.0`，表示关闭，不影响已有实验。启用后，adaptive sampling 会先按原来的失败率公式计算概率，再应用 hard cap 并重新归一化。这样可以保留 hard sample mining，同时防止少数异常或极难 clips 主导训练。
