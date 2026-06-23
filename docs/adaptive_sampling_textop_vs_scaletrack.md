# TextOp 与 ScaleTrack Adaptive Sampling 详细对比

本文对比当前 TextOp 中的 `v2` / `gear_sonic` adaptive sampling，以及 `/data/yzh/ScaleTrack` 中的 motion sampling 机制。重点关注采样粒度、更新信号、概率更新方式、稳定性、覆盖率和大规模数据训练适配性。

## 1. 总览

| 维度 | TextOp v2 | TextOp gear_sonic | ScaleTrack tracking | ScaleTrack interact |
| --- | --- | --- | --- | --- |
| 采样粒度 | motion 级 | motion-time bin 级 | motion 级 | motion 级 yaml 权重 |
| 更新时间 | env reset 时在线更新 | env reset 时在线更新 | 每 `eval_interval` 次 iteration 评估后更新 | 函数存在，但 interact 默认 `eval_during_training=False` |
| 失败信号 | episode termination | episode termination + 当前 time bin | train-set evaluation success/fail | evaluation success/fail |
| 成功信号 | 非 termination 视为 success | 非 termination 视为 success | evaluation 中是否通过阈值 | evaluation 中是否通过阈值 |
| 时间定位 | 无 | 有，按 `gear_sonic_bin_size` 切 bin | 无 | 无 |
| 起始帧采样 | motion 内随机时间 | bin 内采样，可向失败前回退 | motion 内随机 phase | motion 内随机 phase |
| 概率形式 | failure rate + uniform mix | bin failure rate + clipping + uniform mix + bin weight | multiplicative weight update | 重写 motion yaml weight |
| 覆盖保障 | `adaptive_uniform_ratio` | `gear_sonic_uniform_sampling_rate` | weight clamp `[0.03, 1.0]` | weight clamp `[0.03, 1.0]` |
| 多卡同步 | 依赖 DDP 环境内一致统计/日志 | 同上 | rank0 汇总 eval 后 broadcast 权重 | 写 yaml 文件 |

核心判断：

```text
ScaleTrack tracking ADS 比纯 uniform 更有反馈，但仍是 motion-level。
TextOp gear_sonic 的粒度更细，更接近大数据训练里需要的局部困难片段采样。
```

## 2. ScaleTrack tracking 的 ADS 机制

ScaleTrack tracking 初始化时为每条 motion 设置一个权重：

```python
self.motion_sampling_prob = torch.ones(self.num_motion_train, dtype=torch.float)
```

采样时直接按 motion 权重 multinomial：

```python
self.motion_ids[:] = torch.multinomial(
    self.motion_sampling_prob,
    num_samples=self.num_envs,
    replacement=True,
)
```

位置：

```text
/data/yzh/ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/mdp/commands.py
```

### 2.1 更新来源

ScaleTrack 不是在每次 env reset 时根据刚结束的 episode 更新，而是每隔 `eval_interval` 个 PPO iteration 运行一次 evaluation：

```python
if self.eval_during_training and (it + 1) % self.eval_interval == 0:
    eval_dict = self.evaluate_policy()
    eval_test_dict = self.evaluate_policy(test_set=True)
    command.resample_motions()
```

默认配置：

```python
eval_during_training = True
eval_interval = 200
eval_max_steps = 1000
success_discount_coef = 0.999
success_metric_dict = {
    "error_anchor_height": 0.25,
    "error_anchor_rot": 1.0,
}
```

也就是说，ScaleTrack 每 200 iteration 对 train set 做一次评估，根据每条 motion 是否超过成功阈值来更新采样权重。

### 2.2 更新公式

ScaleTrack 的核心更新：

```python
success_discount = math.pow(self.success_discount_coef, self.eval_interval)
new_sampling_prob[failed_idx] /= success_discount
new_sampling_prob[~failed_idx] *= success_discount
new_sampling_prob.clamp_(min=0.03, max=1.0)
```

默认：

```text
success_discount = 0.999 ^ 200 ~= 0.81865
```

所以一次 eval 后：

```text
失败 motion 权重 *= 1 / 0.81865 ~= 1.221
成功 motion 权重 *= 0.81865
权重 clamp 到 [0.03, 1.0]
```

这个机制是 multiplicative reweighting。它会逐渐降低已成功 motion 的权重，提高失败 motion 的相对采样概率。

### 2.3 ScaleTrack 的优点

ScaleTrack 这种方式有几个优点：

1. **更新信号更完整**

   它不是只看 rollout 中当前 env reset 的失败，而是周期性评估 train set motion。理论上每次 eval 都能覆盖整个训练 motion set。

2. **权重更新很温和**

   每次失败只相对提升约 `1.221x`，成功则乘 `0.819x`。这比直接按失败率强采样更平滑。

3. **有硬下限**

   `clamp(min=0.03)` 保证成功 motion 不会完全消失。即使一直成功，也保留一定采样权重。

4. **多卡同步明确**

   分布式时 rank0 汇总 metrics，更新权重，然后 broadcast 到所有 rank。

### 2.4 ScaleTrack 的局限

它的问题也很明确：

1. **motion 级粒度过粗**

   一条 motion 里可能只有 1 秒很难，但 ScaleTrack 会提高整条 motion 的权重。

   ```text
   一个 jump motion 只有落地失败 -> 整条 jump motion 都被过采样
   一个 carry motion 只有抓取切换失败 -> 整条 carry motion 都被过采样
   ```

2. **不能定位失败时间段**

   `_resample_command` 里训练时仍然是：

   ```python
   phase = torch.rand(...)
   self.time_steps = phase * (motion_len - 1)
   ```

   也就是在 motion 内随机起始时间。失败发生在哪里，不会反馈到下次采样的时间点。

3. **评估成本高**

   每 `eval_interval=200` iteration 做一次 train-set evaluation。数据量上到 155h 或 700h 后，完整 train set eval 会变得非常贵。

4. **阈值依赖强**

   成败只由 `success_metric_dict` 决定。当前 tracking 默认主要看：

   ```text
   error_anchor_height > 0.25
   error_anchor_rot > 1.0
   ```

   如果 body、joint、EE、object tracking 已经很差但 anchor 还没超阈值，这条 motion 仍可能被视为成功。

5. **没有显式 failure rate**

   它记录的是当前 eval 成败，然后对权重做乘法更新；不是统计长期 `num_failures / num_episodes`。

## 3. TextOp v2 的 ADS 机制

TextOp v2 使用 motion-level 失败率：

```python
p_fail = failed_motion_count / (failed_motion_count + success_motion_count + 1e-8)
p_fail_sample = p_fail ** adaptive_beta
sampling_probabilities = (
    p_fail_sample * (1 - adaptive_uniform_ratio)
    + adaptive_uniform_ratio / num_motion
)
```

统计量通过 EMA 更新：

```python
failed_motion_count = alpha * current_failed + (1 - alpha) * failed_motion_count
success_motion_count = alpha * current_success + (1 - alpha) * success_motion_count
```

### 3.1 TextOp v2 的优点

1. **在线更新，不需要完整 eval**

   每次 env reset 都能根据实际训练中的 termination 更新统计。

2. **有 failure rate**

   比只看 raw failure count 更合理。

3. **有 uniform mix**

   `adaptive_uniform_ratio` 能保留全数据覆盖。

4. **有 `max_prob_over_uniform`**

   可以直接限制 motion-level top probability，避免单条 motion 过度集中。

### 3.2 TextOp v2 的局限

1. **仍然是 motion-level**

   和 ScaleTrack 一样，v2 不知道失败发生在 motion 哪个时间段。

2. **早期统计噪声影响较大**

   如果某些 motion 早期被采到且失败多，可能很快获得较高概率。需要靠 `adaptive_alpha`、`adaptive_beta`、`adaptive_uniform_ratio` 和 `max_prob_over_uniform` 控制。

3. **success 定义比较宽**

   当前实现里：

   ```python
   episode_success = ~episode_failed
   ```

   也就是 timeout 和 motion clip 正常结束都算 success。这个定义对稳定训练有利，但不是严格 tracking quality success。

## 4. TextOp gear_sonic 的 ADS 机制

TextOp gear_sonic 是 bin-level：

```python
motion -> 按 gear_sonic_bin_size 切成多个时间 bin
```

每个 bin 维护：

```text
gear_sonic_num_failures
gear_sonic_num_episodes
gear_sonic_failure_rate = num_failures / num_episodes
```

采样概率：

```python
failure_rate_clipped = clamp(failure_rate, 0, mean(failure_rate) * max_over_mean)
failure_based_prob = failure_rate_clipped / sum(failure_rate_clipped)
sampling_prob = failure_based_prob * (1 - uniform_rate) + uniform_prob * uniform_rate
sampling_prob *= gear_sonic_bin_weights
sampling_prob /= sampling_prob.sum()
```

采样后：

```python
sampled_bin -> bin 内随机 time_step
if pre_failure_sample_window > 0:
    time_step -= random offset
```

### 4.1 gear_sonic 的优点

1. **局部困难片段定位**

   它不会因为某条 motion 的局部失败而提高整条 motion 的采样概率。

2. **失败前窗口更适合控制学习**

   失败往往不是失败帧才产生的，而是在失败前几十到几百步逐渐进入错误状态。`pre_failure_sample_window` 可以训练策略避免进入失败轨迹。

3. **failure rate + clipping**

   比 ScaleTrack 的乘法权重更直接地估计 bin 难度，同时 clipping 可以抑制异常片段。

4. **uniform mix 保留覆盖**

   `gear_sonic_uniform_sampling_rate` 保证全数据仍然被采样。

5. **更适合大数据**

   对 155h 或 700h 数据，motion-level 太粗，bin-level 可以更有效使用训练预算。

### 4.2 gear_sonic 的风险

1. **实现更复杂**

   需要维护 bin、failure rate、episode count、bin weight、metrics。

2. **调参更敏感**

   如果 `uniform_sampling_rate` 太低，或者 `failure_rate_max_over_mean` 太高，可能集中到 jump / run / turn 这类高失败片段。

3. **当前 TextOp 实现中 uniform 后又乘了 bin weight**

   所以实际最低概率不严格等于 `uniform_rate / num_bins`。这点需要监控 `sampling_entropy` 和 `sampling_top1_prob`。

## 5. 三者谁更激进

从“采样分布偏向困难样本”的角度：

```text
TextOp gear_sonic 可最精确，也可能最强
TextOp v2 中等，取决于 beta / uniform / max_prob
ScaleTrack tracking 最温和，但粒度最粗
```

更具体地说：

| 方法 | 激进程度 | 原因 |
| --- | --- | --- |
| ScaleTrack tracking | 温和 | 每 200 iter 才 eval 更新一次；单次失败权重约 `+22%`；权重 clamp `[0.03,1.0]` |
| TextOp v2 | 中等 | reset 在线更新；按 motion failure rate 采样；可用 beta 放大；有 uniform 和 max cap |
| TextOp gear_sonic | 中等到强 | reset 在线更新；bin-level；困难时间段会被持续加强；有 pre-failure window |

但“激进”不等于“不好”。gear_sonic 的优势是它把激进程度用在局部困难片段，而不是整条 motion。

## 6. 谁更适合大数据长期训练

### ScaleTrack tracking

适合：

```text
中等规模数据
需要稳定、简单、低风险的 motion-level curriculum
训练中可以承担周期性完整 evaluation 成本
```

不适合：

```text
700h 超大数据
一条 motion 内部难度差异很大
希望快速定位失败时间段
```

### TextOp v2

适合：

```text
想要在线 feedback
不想做完整 train-set eval
数据规模中等到较大
motion 内部难度差异不是特别大
```

风险：

```text
motion-level 粒度仍然偏粗
如果 max_prob_over_uniform 没设，可能局部 motion 过采样
```

### TextOp gear_sonic

适合：

```text
155h / 700h 大规模 motion 数据
motion 内部存在明显局部困难片段
希望稳定训练同时提高困难片段学习效率
```

风险：

```text
需要监控 sampling_entropy / sampling_top1_prob
需要合理设置 uniform_sampling_rate 和 failure_rate_max_over_mean
```

## 7. 对 TextOp 当前训练的建议

如果目标是“保持稳定，同时尽可能覆盖动作”，建议优先继续使用 TextOp gear_sonic，而不是迁移 ScaleTrack 的 ADS。

推荐参数：

```yaml
env.commands.motion.ads_type: gear_sonic
env.commands.motion.gear_sonic_bin_size: 50
env.commands.motion.gear_sonic_uniform_sampling_rate: 0.2
env.commands.motion.gear_sonic_failure_rate_max_over_mean: 100.0
env.commands.motion.gear_sonic_pre_failure_sample_window: 200
```

如果发现 sampling entropy 低于 `0.90`，或者 top1/top100 bin 明显集中，可以改成：

```yaml
env.commands.motion.gear_sonic_uniform_sampling_rate: 0.25
env.commands.motion.gear_sonic_failure_rate_max_over_mean: 50.0
```

如果训练稳定性下降，困难动作学不住，可以退回：

```yaml
env.commands.motion.gear_sonic_uniform_sampling_rate: 0.15
env.commands.motion.gear_sonic_failure_rate_max_over_mean: 100.0
```

## 8. 可以从 ScaleTrack 借鉴什么

虽然不建议直接用 ScaleTrack ADS 替代 gear_sonic，但有几件事值得借鉴：

1. **周期性全量评估作为校准信号**

   TextOp 可以保留 gear_sonic 在线 ADS，同时每隔一段时间跑一次 train subset evaluation，检查 ADS 是否把某些 motion 判错。

2. **基于 tracking metric 的 success/fail**

   当前 TextOp ADS 主要看 termination。可以增加可选模式，让 success/fail 也参考 episode 内最大 tracking error，例如 anchor/body/EE error。

3. **显式记录 motion-level coverage**

   ScaleTrack 的评估天然知道哪些 motion 成功失败。TextOp gear_sonic 可以额外记录每个 motion 的采样次数、成功率、最后采样 iteration，避免长尾 motion 长期没有被访问。

4. **权重下限思想**

   Gear_sonic 有 uniform mix，但由于乘了 bin weight，实际最低概率不严格等于 uniform 下限。可以考虑增加一个真正的 post-weight probability floor。

## 9. 最终判断

ScaleTrack 的 ADS 是：

```text
周期性 evaluation -> motion 成败 -> 乘法更新 motion 权重
```

TextOp v2 是：

```text
在线 episode reset -> motion failure rate EMA -> motion 概率采样
```

TextOp gear_sonic 是：

```text
在线 episode reset -> motion-time bin failure rate -> bin 概率采样 -> 失败前窗口
```

对于当前 TextOp 的目标，尤其是大数据、长时间、动作覆盖和稳定性同时要求较高的训练，推荐优先路线是：

```text
TextOp gear_sonic
+ 更高 uniform rate
+ 更低 failure_rate_max_over_mean
+ 增加 motion-level coverage 监控
+ 可选周期性 eval 校准
```

而不是回到 ScaleTrack 的纯 motion-level ADS。
