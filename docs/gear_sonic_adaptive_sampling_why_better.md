# Gear-Sonic Adaptive Sampling 为什么更适合大数据稳定训练

本文总结 `ads_type=gear_sonic` 相比当前 TextOp v2 adaptive sampling 的核心优势，以及它为什么更适合 155h 到 700h 级别的大规模 motion tracking 训练。

## 核心结论

Gear-Sonic ADS 的优势不是“更激进”，而是更稳、更局部、更适合长时间大数据训练。

v2 更像：

```text
哪条 motion 难，就多采哪条 motion。
```

Gear-Sonic 更像：

```text
哪一小段时间导致失败，就多采失败前后这一段，同时保留全数据覆盖。
```

这对 motion tracking 很重要，因为一条 motion 内通常只有少数片段真正困难，例如起跳、落地、急转、抓取、搬运、坐下起身、快速启停。把整条 motion 都提高采样概率会浪费训练预算，也更容易让数据分布偏掉。

## 1. 时间 bin 粒度比 motion 粒度更合理

v2 主要是 motion-level adaptive sampling：

```text
某个 motion 失败多 -> 整条 motion 的概率升高
```

Gear-Sonic 使用 bin-level adaptive sampling：

```text
某个 motion 的某一段失败多 -> 只提高该时间段附近的采样概率
```

当前实现中：

```yaml
gear_sonic_bin_size: 50
```

也就是每条 motion 按 50 帧切成多个 bin。这样对于一条很长的 motion，ADS 不会因为某个短暂困难片段而把整条 motion 都过采样。

## 2. 失败前窗口采样更符合控制学习

Gear-Sonic 不只是从失败点重新采样，而是使用失败前窗口：

```yaml
gear_sonic_pre_failure_sample_window: 200
```

控制任务里，失败往往不是失败那一帧突然发生的。策略通常在失败前几十到几百步已经进入了错误状态，例如：

```text
起跳前姿态不对 -> 空中身体偏转 -> 落地失败
急转前速度过大 -> 脚步跟不上 -> 姿态崩掉
抓取前手部轨迹偏移 -> 后续 EE tracking 失败
```

因此从失败点前一段时间重采样，比只采失败帧更有效。它训练的是“如何避免进入失败轨迹”，而不是只训练“失败之后怎么补救”。

## 3. 保留 uniform sampling，减少遗忘

Gear-Sonic 有显式的均匀采样比例：

```yaml
gear_sonic_uniform_sampling_rate: 0.1
```

这保证了即使某些困难动作被提高采样概率，普通动作仍然会持续进入训练。

这一点对大数据尤其重要。如果没有 uniform 成分，采样会逐渐集中到 jump、turn、jog、run 这些高失败率动作，普通 walk、idle、reach、carry、object interaction 可能被长期低采样，导致策略后期遗忘或覆盖不足。

## 4. 使用 failure rate，而不是 raw failure count

更合理的难度估计应该接近：

```text
failure_rate = num_failures / num_episodes
```

而不是只看：

```text
num_failures
```

只看失败次数会偏向被采样次数多的 motion 或长 motion。Gear-Sonic 使用 failure rate，可以更接近“这个 bin 本身有多难”，而不是“它被采过多少次”。

## 5. Clipping 防止异常片段支配训练

Gear-Sonic 有 failure rate clipping：

```yaml
gear_sonic_failure_rate_max_over_mean: 200.0
```

这个参数限制的是：

```text
failure_rate <= mean_failure_rate * gear_sonic_failure_rate_max_over_mean
```

它不是直接限制：

```text
sampling_prob <= uniform_prob * 200
```

所以当日志里出现：

```text
Metrics/motion/sampling_top1_prob = 0.0002
```

不能理解成“200 倍 uniform”。例如当前 run 有：

```text
num_bins = 69405
uniform_bin_prob = 1 / 69405 = 0.0000144
top1_prob = 0.0002
top1_prob / uniform_bin_prob ~= 14
```

也就是说当前 top1 bin 约为 14 倍均匀采样概率，而不是 200 倍。

## 6. 当前 run 的采样状态

以该 run 为例：

```text
/data/yzh/TextOp/TextOpTracker/logs/rsl_rl/rgz_loco_lomanip_object_transformer_vae_best_1_ddp/2026-06-10_16-19-22_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug
```

`model_1000-adpsam_count.pkl` 的采样分布大致为：

| 指标 | 数值 | 解释 |
| --- | ---: | --- |
| bin 数 | 69,405 | 按 50 帧切分后的采样单元 |
| motion 数 | 9,933 | 当前训练使用的 motion 数 |
| sampling entropy | 0.931 | 有明显困难样本偏置，但没有塌缩 |
| top1 bin / uniform | 约 15x | 最困难 bin 是均匀概率的约 15 倍 |
| top100 bin mass | 约 1.97% | top bin 没有支配训练 |
| top1000 bin mass | 约 13.05% | 有温和到中等的困难片段偏置 |
| top10000 bin mass | 约 57.96% | 前 14.4% 的 bin 吃掉约 58% 概率 |
| top10 motion mass | 约 0.24% | motion 级别没有严重集中 |
| top1000 motion mass | 约 18.1% | motion 级别存在温和偏置 |

高概率 bin 主要集中在：

```text
jump_ff_*
turn_jump_*
jog_ff_loop_*
jog_ff_stop_*
run_loop_*
```

这说明 ADS 正在把预算给真正容易失败的快速移动、跳跃、急转和启停片段，而不是无差别提高整条 motion 的概率。

## 7. 对稳定性和覆盖率的推荐参数

如果目标是保持稳定，同时提高动作覆盖，建议比当前更温和一些：

```yaml
env.commands.motion.ads_type: gear_sonic
env.commands.motion.gear_sonic_bin_size: 50
env.commands.motion.gear_sonic_uniform_sampling_rate: 0.2
env.commands.motion.gear_sonic_failure_rate_max_over_mean: 100.0
env.commands.motion.gear_sonic_pre_failure_sample_window: 200
```

如果更偏覆盖，可以进一步使用：

```yaml
env.commands.motion.gear_sonic_uniform_sampling_rate: 0.25
env.commands.motion.gear_sonic_failure_rate_max_over_mean: 50.0
```

不建议一开始把 uniform rate 提到 `0.5`。那会让 ADS 接近半均匀采样，困难动作的复训强度会明显下降，jump、turn、jog、run 这些动作可能学不牢。

## 8. 建议监控指标

训练中建议重点看：

```text
Metrics/motion/sampling_entropy
Metrics/motion/sampling_top1_prob
Metrics/motion/pfail_total
Episode_Termination/time_out
Episode_Termination/ee_body_pos
Train/mean_episode_length
Train/mean_reward
```

经验上：

```text
sampling_entropy > 0.94     更偏覆盖
sampling_entropy 0.92~0.94  中等困难样本偏置
sampling_entropy < 0.90     需要检查是否过度集中
```

同时：

```text
Train/mean_episode_length 应尽量保持接近 episode 上限
Episode_Termination/time_out 应保持较高
Episode_Termination/ee_body_pos 不应重新升高
```

如果 `sampling_top1_prob` 持续升高，同时 entropy 持续下降，说明采样正在向少数困难 bin 集中。此时应提高 `gear_sonic_uniform_sampling_rate` 或降低 `gear_sonic_failure_rate_max_over_mean`。

## 总结

Gear-Sonic ADS 更适合大数据长期训练的原因是：

```text
局部困难片段加权
+ 失败前窗口重采样
+ failure rate 归一化
+ clipping 抑制异常片段
+ uniform sampling 保留全数据覆盖
```

它的目标不是把训练变得更激进，而是在大规模 motion 数据上，把有限的训练预算更稳定地分配给真正困难的局部片段，同时避免忘掉长尾和简单动作。
