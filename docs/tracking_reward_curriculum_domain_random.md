# Tracking 任务：Reward / Curriculum / Domain Random 总结（TextOpTracker）

> 代码位置基于：`TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/`

## 1. Reward（奖励项）

Reward 配置入口在 `tracking_env_cfg.py` 的 `RewardsCfg`，具体计算函数主要在 `mdp/rewards.py`（部分来自 IsaacLab 内置 `mdp`）。

### 1.1 运动跟踪（主正奖励，指数型误差）

以下项的共同形式是 **exp(-error / std^2)**，误差越小，奖励越接近 1。

- `motion_global_anchor_pos`（weight = **+0.5**，std = 0.3）
  - **含义**：全局 anchor 位置跟踪
  - **计算**：\sum (anchorposw - robotanchorposw)^2
  - **实现**：`mdp/rewards.py::motion_global_anchor_position_error_exp`
- `motion_global_anchor_ori`（weight = **+0.5**，std = 0.4）
  - **含义**：全局 anchor 姿态跟踪
  - **计算**：`quat_error_magnitude(anchor_quat_w, robot_anchor_quat_w)^2`
  - **实现**：`mdp/rewards.py::motion_global_anchor_orientation_error_exp`
- `motion_body_pos`（weight = **+1.0**，std = 0.3）
  - **含义**：各 body 的相对位置跟踪（对 body 取 mean）
  - **计算**：对选定 body：\sum (bodyposrelativew - robotbodyposw)^2，再对 body 求 mean
  - **实现**：`mdp/rewards.py::motion_relative_body_position_error_exp`
- `motion_body_ori`（weight = **+1.0**，std = 0.4）
  - **含义**：各 body 的相对姿态跟踪（对 body 取 mean）
  - **计算**：对选定 body：`quat_error_magnitude(body_quat_relative_w, robot_body_quat_w)^2`，再对 body 求 mean
  - **实现**：`mdp/rewards.py::motion_relative_body_orientation_error_exp`
- `motion_body_lin_vel`（weight = **+1.0**，std = 1.0）
  - **含义**：各 body 的全局线速度跟踪（对 body 取 mean）
  - **实现**：`mdp/rewards.py::motion_global_body_linear_velocity_error_exp`
- `motion_body_ang_vel`（weight = **+1.0**，std = 3.14）
  - **含义**：各 body 的全局角速度跟踪（对 body 取 mean）
  - **实现**：`mdp/rewards.py::motion_global_body_angular_velocity_error_exp`

### 1.2 动作/约束惩罚（负奖励）

- `action_rate_l2`（weight = **-1e-1**）
  - **含义**：动作变化率（\Delta a）的 L2 惩罚，鼓励平滑控制
  - **实现**：IsaacLab 内置 `mdp.action_rate_l2`
- `joint_limit`（weight = **-10.0**）
  - **含义**：关节位置接近/超过限制的惩罚（所有关节 `".*"`）
  - **实现**：IsaacLab 内置 `mdp.joint_pos_limits`
- `undesired_contacts`（weight = **-0.1**，threshold = 1.0）
  - **含义**：不希望发生的 body 接触惩罚
  - **传感器**：`contact_forces`
  - **body 选择**：使用正则“排除列表”，把以下 link 排除在惩罚之外：
    - `left_ankle_roll_link`, `right_ankle_roll_link`, `left_wrist_yaw_link`, `right_wrist_yaw_link`
  - **实现**：IsaacLab 内置 `mdp.undesired_contacts`

### 1.3 接触/落地/安全惩罚（带 pfail 门控）

本代码里有一类“门控奖励/惩罚”：当

- `pfail_total < pfail_threshold`

时才计算；否则该项直接为 0（不影响总 reward）。

`pfail_total` 来自 motion command：`env.command_manager.get_term("motion").metrics["pfail_total"]`。

- `feet_force`（weight = **-0.0**，threshold = 600，pfail_threshold = 1.0）
  - **含义**：足部接触力惩罚（当前 weight 为 -0.0 等于关闭）
  - **body**：`".*ankle_roll.*"`
  - **实现**：`mdp/rewards.py::contact_forces_cond_on_pfail`（内部调用 IsaacLab `contact_forces`）
- `feet_slide`（weight = **-0.1**，pfail_threshold = 0.2）
  - **含义**：接触时的脚底水平滑动速度惩罚
  - **接触判定**：接触力 history 的最大范数 > 1.0
  - **惩罚**：\sum ||v_{xy}||（只取 xy 速度）
  - **实现**：`mdp/rewards.py::feet_slide_cond_on_pfail`（门控） + `mdp/rewards.py::feet_slide`
- `soft_landing`（weight = **-1e-5**，pfail_threshold = 0.2）
  - **含义**：鼓励“轻柔落地”，惩罚落地瞬间冲击
  - **惩罚**：落地瞬间 `first_contact` 时的 `||net_forces_w||` 求和
  - **实现**：`mdp/rewards.py::soft_landing_cond_on_pfail`（门控） + `mdp/rewards.py::soft_landing`
- `overspeed`（weight = **-0.1**，max_velocity = 20.0，pfail_threshold = 0.15）
  - **含义**：关节速度超过手动阈值的“超速关节数量”惩罚
  - **实现**：`mdp/rewards.py::joint_vel_out_of_manual_limit_cond_on_pfail_reward`
- `overeffort`（weight = **-0.1**，pfail_threshold = 0.15）
  - **含义**：关节力矩被限幅/饱和的“关节数量”惩罚
  - **判定**：`computed_torque` 与 `applied_torque` 不相等（`isclose` 取反）
  - **实现**：`mdp/rewards.py::joint_effort_out_of_limit_fixed_cond_on_pfail_reward`

## 2. Curriculum（课程学习）

`tracking_env_cfg.py` 中存在 `CurriculumCfg`，但当前实现是空的：

- `class CurriculumCfg: pass`
- `TrackingEnvCfg.curriculum: CurriculumCfg = CurriculumCfg()`

因此 **这份 Tracking 配置里没有启用任何课程学习项**（没有随训练进度自动调权重/难度的 term）。

## 3. Domain Randomization（域随机化）

域随机化主要通过 `tracking_env_cfg.py` 的 `EventCfg` 实现（IsaacLab 的事件系统），分为：

- **startup**：环境初始化时执行一次
- **interval**：每隔一段随机时间执行

### 3.1 Startup 随机化（初始化阶段）

- `physics_material`（startup）
  - **内容**：随机化刚体材质（摩擦/回弹），并做 bucket 化
  - **参数**：
    - `static_friction_range`: (0.3, 1.6)
    - `dynamic_friction_range`: (0.3, 1.2)
    - `restitution_range`: (0.0, 0.5)
    - `num_buckets`: 64
  - **作用对象**：`SceneEntityCfg("robot", body_names=".*")`
  - **实现**：`mdp.randomize_rigid_body_material`（来自 IsaacLab）
- `add_joint_default_pos`（startup）
  - **内容**：随机化关节 default position（模拟标定误差）
  - **参数**：
    - `pos_distribution_params`: (-0.01, 0.01)
    - `operation`: "add"
  - **作用对象**：`SceneEntityCfg("robot", joint_names=[".*"])`
  - **实现**：`mdp/events.py::randomize_joint_default_pos`
    - 同时会更新 action term `joint_pos` 的 offset（因为它不会自动同步）
- `base_com`（startup）
  - **内容**：随机化 torso（`torso_link`）的质心 CoM
  - **参数**：
    - x: (-0.025, 0.025)
    - y: (-0.05, 0.05)
    - z: (-0.05, 0.05)
  - **作用对象**：`SceneEntityCfg("robot", body_names="torso_link")`
  - **实现**：`mdp/events.py::randomize_rigid_body_com`
    - 注意：该实现使用 CPU 张量设置 CoM，设计上建议只在初始化阶段使用

### 3.2 Interval 随机化（训练过程中周期触发）

- `push_robot`（interval）
  - **内容**：随机推机器人（通过直接设置速度）
  - **触发间隔**：`interval_range_s = (1.0, 3.0)`
  - **速度范围**：使用 `VELOCITY_RANGE`：
    - x: (-0.5, 0.5), y: (-0.5, 0.5), z: (-0.2, 0.2)
    - roll: (-0.52, 0.52), pitch: (-0.52, 0.52), yaw: (-0.78, 0.78)
  - **实现**：`mdp.push_by_setting_velocity`（来自 IsaacLab）

---

## 4. 关键结论（快速扫读版）

- **Reward**：以 motion tracking 的 6 个 exp(-error/std^2) 正奖励为主；再叠加动作平滑、关节限制、非期望接触等惩罚；部分惩罚项在 `pfail_total` 过大时会被门控为 0。
- **Curriculum**：当前 Tracking 配置里 **未启用**（`CurriculumCfg` 为空）。
- **Domain Random**：主要在 `EventCfg`，startup 随机材质/关节零位/torso 质心，interval 随机 push（扰动速度）。

