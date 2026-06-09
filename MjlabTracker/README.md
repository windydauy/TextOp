# MjlabTracker

`MjlabTracker` 是 `latent_tracker` 在 mjlab 原生环境上的 G1 motion tracking 迁移版本。它复用 mjlab 的 `ManagerBasedRlEnv`、MuJoCo/Warp 仿真和 RSL-RL 训练入口，提供一组 `Mjlab-LatentTracker-Flat-G1...` tracking 任务。

当前仓库路径：

```bash
/home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
```

## 环境

推荐使用已有 conda 环境：

```bash
conda activate latent_tracker
```

如果没有安装 editable package，可以临时设置源码路径：

```bash
export PYTHONPATH=/home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker/source/latent_tracker:/home/humanoid/yzh/mjlab/src:$PYTHONPATH
```

长期开发建议安装：

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
python -m pip install -e .
```

## 快速验证

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
python -m py_compile $(find source scripts tests -name "*.py")
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

如果 pytest 被系统插件污染导致缺包，保留 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。

## 查看任务

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
python - <<'PY'
import latent_tracker  # registers tasks
from mjlab.tasks.registry import list_tasks

for task in list_tasks():
    if task.startswith("Mjlab-LatentTracker-Flat-G1"):
        print(task)
PY
```

常用任务：

```text
Mjlab-LatentTracker-Flat-G1
Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-NMMLP
Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-MotionAE-NMMLP
Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-TransformerVAE
Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-TransformerVAE-NMMLP
Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-NMMLP
Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-SplitBodyReward-NMMLP
```

## 数据

tracking command 默认读取 motion npz。`motion_files` 现在支持两种形式：

```bash
# 单个文件
--env.commands.motion.motion-files '["/path/to/motion.npz"]'

# 文件夹，内部会递归查找所有 motion.npz
--env.commands.motion.motion-files '["/path/to/motion_folder"]'
```

motion npz 需要包含训练使用的字段，例如：

```text
fps
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
body_names
joint_names
```

`body_names` 和 `joint_names` 存在时，loader 会按当前 G1 MuJoCo robot 的 body/joint 名称重排数据。

## 训练

基础训练入口：

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
python scripts/rsl_rl/train.py Mjlab-LatentTracker-Flat-G1 \
  --env.commands.motion.motion-files '["/home/humanoid/yzh/TextOp/motion_accloss2.npz"]' \
  --env.scene.num-envs 4096 \
  --gpu-ids '[0]'
```

TransformerVAE + folder 数据示例：

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab
bash scripts/train_mjlab_transformer_vae_g1_before_2023.sh
```

该脚本使用：

```text
task: Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-TransformerVAE
data: /home/humanoid/yzh/TextOp/g1_before_2023
conda env: latent_tracker
```

并通过 config override 关闭：

```text
env.commands.motion.freeze_frame_aug = False
env.commands.motion.freeze_frame_aug_prob = 0.0
env.events.randomize_rigid_body_mass.params.ranges = (1.0, 1.0)
```

常用覆盖：

```bash
NUM_ENVS=2048 MAX_ITERATIONS=10000 DEVICE=cuda:1 \
bash scripts/train_mjlab_transformer_vae_g1_before_2023.sh
```

## 播放

用 policy/env 播放 tracking：

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
python scripts/rsl_rl/play.py Mjlab-LatentTracker-Flat-G1 \
  --motion-file /home/humanoid/yzh/TextOp/motion_accloss2.npz \
  --num-envs 1 \
  --device cuda:0
```

纯 npz replay 可用：

```bash
cd /home/humanoid/yzh/LatentTrackerMjlab/MjlabTracker
python scripts/replay_motion_npz_viser.py \
  /home/humanoid/yzh/TextOp/motion_accloss2.npz \
  --port 8080 \
  --loop
```

然后打开：

```text
http://127.0.0.1:8080
```

## 目录结构

```text
source/latent_tracker/
  latent_tracker/robots/g1/                      G1 robot asset/config
  latent_tracker/tasks/tracking/                 tracking MDP、reward、observation、command
  latent_tracker/tasks/tracking/config/g1/       G1 task registration and env factories

scripts/rsl_rl/
  train.py                                       mjlab-native RSL-RL training entry
  play.py                                        playback/evaluation entry
  export.py                                      export helper

scripts/replay_motion_npz_viser.py               pure motion npz viewer
tests/test_mjlab_contracts.py                    wiring and compatibility tests
```

## 常见问题

`ModuleNotFoundError: latent_tracker`

设置 `PYTHONPATH`，或在 `MjlabTracker` 下执行 `python -m pip install -e .`。

`ModuleNotFoundError: mjlab`

确认 `/home/humanoid/yzh/mjlab/src` 在 `PYTHONPATH`，或 mjlab 已经安装到当前环境。

第一次运行很慢

MuJoCo Warp 第一次会编译 kernel，后续会走缓存。

folder 数据没读到

传入目录时只递归收集文件名为 `motion.npz` 的文件；如果是单文件路径，则可以是任意 npz 文件名。
