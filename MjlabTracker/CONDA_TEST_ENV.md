# MjlabTracker Conda 测试环境

本文档用于给 `/home/humanoid/yzh/TextOp/MjlabTracker` 创建一个独立 Conda 环境并运行迁移后的 mjlab-native G1 tracking 测试。

## 推荐选择

如果只是立刻验证当前迁移结果，可以直接用已有环境：

```bash
export PYTHONPATH=/home/humanoid/yzh/TextOp/MjlabTracker/source/latent_tracker:/home/humanoid/yzh/mjlab/src
/home/humanoid/yzh/mjlab/.venv/bin/python -m pytest -q /home/humanoid/yzh/TextOp/MjlabTracker/tests
```

如果要长期开发，建议使用下面的 Conda 环境。

## 1. 创建 Conda 环境

```bash
conda create -n mjlab-tracker python=3.12 -y
conda activate mjlab-tracker
python -m pip install -U pip uv
```

## 2. 安装 mjlab

mjlab 的 `pyproject.toml` 配置了 PyTorch、NVIDIA、MuJoCo 等 uv source。请在 mjlab 仓库目录内安装：

```bash
cd /home/humanoid/yzh/mjlab
uv pip install --system -e ".[cu128]" --prerelease allow
```

当前 `mujoco-warp 3.6.0` 需要 MuJoCo Python wheel 暴露 `mjENBL_MULTICCD`。如果 `uv` 解析到不兼容的 `mujoco`，可能会报：

```text
AttributeError: type object 'mujoco._enums.mjtEnableBit' has no attribute 'mjENBL_MULTICCD'
```

在 `mjlab-tracker` 环境里执行下面的检查：

```bash
python - <<'PY'
import mujoco
print("mujoco", mujoco.__version__)
print("has mjENBL_MULTICCD:", hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"))
PY
```

如果 `has mjENBL_MULTICCD` 是 `False`，再安装当前 MuJoCo nightly 源里可用的 dev wheel。注意不要写死已经下架的 build 号，例如 `3.7.0.dev895794022` 当前不可用；下面这个版本已在本机验证可安装：

```bash
python -m pip install --force-reinstall --no-deps -i https://py.mujoco.org --pre mujoco==3.7.0.dev896346482
python -m pip install scipy
```

如果使用的是 mjlab 自带的 uv `.venv`，里面可能默认没有 `pip`。可以先补 pip：

```bash
/home/humanoid/yzh/mjlab/.venv/bin/python -m ensurepip --upgrade
```

如果机器没有 CUDA 或只想跑 CPU 烟测，可以改用：

```bash
cd /home/humanoid/yzh/mjlab
uv pip install --system -e ".[cpu]" --prerelease allow
```

## 3. 安装 MjlabTracker

```bash
cd /home/humanoid/yzh/TextOp/MjlabTracker
python -m pip install -e .
```

## 4. 设置可选路径

正常 `pip install -e .` 后不需要手动设置 `PYTHONPATH`。如果你不安装包、只想临时测试源码，可以这样设置：

```bash
export PYTHONPATH=/home/humanoid/yzh/TextOp/MjlabTracker/source/latent_tracker:/home/humanoid/yzh/mjlab/src:$PYTHONPATH
```

## 5. 快速验证

```bash
cd /home/humanoid/yzh/TextOp/MjlabTracker
python -m py_compile $(find source scripts tests -name "*.py")
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

预期结果：

```text
4 passed
```

## 6. 最小 create/reset/step 烟测

```bash
python - <<'PY'
import torch
from mjlab.envs import ManagerBasedRlEnv
from latent_tracker.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

cfg = G1FlatEnvCfg()
cfg.scene.num_envs = 1

env = ManagerBasedRlEnv(cfg, device="cpu")
obs, extras = env.reset()
actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
obs, rew, terminated, truncated, extras = env.step(actions)

print("action_dim", env.action_manager.total_action_dim)
print("actor_shape", tuple(obs["actor"].shape))
print("critic_shape", tuple(obs["critic"].shape))
print("rew_shape", tuple(rew.shape))
print("terminated", terminated.tolist())
print("truncated", truncated.tolist())
env.close()
PY
```

预期关键输出：

```text
action_dim 29
actor_shape (1, 428)
critic_shape (1, 554)
rew_shape (1,)
```

第一次运行会编译 MuJoCo Warp kernels，可能需要几十秒；之后会走缓存。

## 7. 查看已注册任务

```bash
python - <<'PY'
import latent_tracker  # registers tasks
from mjlab.tasks.registry import list_tasks

for task in list_tasks():
    if task.startswith("Mjlab-LatentTracker-Flat-G1"):
        print(task)
PY
```

应看到 17 个 `Mjlab-LatentTracker-Flat-G1...` 任务。

## 8. 训练 / 播放入口

训练入口：

```bash
cd /home/humanoid/yzh/TextOp/MjlabTracker
python scripts/rsl_rl/train.py Mjlab-LatentTracker-Flat-G1
```

播放入口：

```bash
cd /home/humanoid/yzh/TextOp/MjlabTracker
python scripts/rsl_rl/play.py Mjlab-LatentTracker-Flat-G1 --device cuda:0
```

如果只做 CPU wiring 验证，把 `--device cuda:0` 改成 `--device cpu`。

## 9. 常见问题

- `ModuleNotFoundError: tyro`：说明没有使用 mjlab 环境或 mjlab 依赖没装完整，重新执行第 2 步。
- `ModuleNotFoundError: latent_tracker`：执行第 3 步，或设置第 4 步的 `PYTHONPATH`。
- `mjENBL_MULTICCD` 报错：先执行第 2 步里的检查；如果缺失，再安装当前可用的 MuJoCo dev wheel。
- `ModuleNotFoundError: scipy`：执行 `python -m pip install scipy`。
- `pytest` 加载 ROS 插件后报 `ModuleNotFoundError: lark`：使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests`。
- 第一次运行卡在 `Warp initialized` / kernel compile：这是正常的 MuJoCo Warp 编译缓存过程。
- `motion file` 相关报错：默认使用 `/home/humanoid/yzh/TextOp/motion_accloss2.npz`，请确认该文件存在。

1. 纯 replay npz 本身

  我给你加了脚本：

  MjlabTracker/scripts/replay_motion_npz_viser.py

  它会直接读取 motion.npz，逐帧把：

  body_pos_w/body_quat_w 的 pelvis
  joint_pos

  写进 MuJoCo，然后用 Viser 显示。它会按 joint_names 自动重排到 MuJoCo XML 的关节顺
  序。

  运行：

  cd /home/humanoid/yzh/TextOp/MjlabTracker

  PYTHONPATH=/home/humanoid/yzh/TextOp/MjlabTracker/source/latent_tracker:/home/
  humanoid/yzh/mjlab/src \
  /home/humanoid/miniconda3/envs/mjlab-tracker/bin/python scripts/
replay_motion_npz_viser.py \
    /home/humanoid/yzh/TextOp/motion_accloss2.npz \
    --port 8080 \
    --loop

  然后浏览器打开：

  http://127.0.0.1:8080

  如果你要 replay 其他 npz，把路径换掉：

  ... scripts/replay_motion_npz_viser.py /path/to/motion.npz --port 8080 --loop

  2. 用 mjlab play 看 policy tracking

  这个不是纯 replay reference，而是让 agent 去跟踪 motion：

  cd /home/humanoid/yzh/TextOp/MjlabTracker

  PYTHONPATH=/home/humanoid/yzh/TextOp/MjlabTracker/source/latent_tracker:/home/
  humanoid/yzh/mjlab/src \
  /home/humanoid/miniconda3/envs/mjlab-tracker/bin/python scripts/rsl_rl/play.py \
    Mjlab-LatentTracker-Flat-G1 \
    --agent zero \
    --motion-file /home/humanoid/yzh/TextOp/motion_accloss2.npz \
    --viewer viser \
    --num-envs 1 \
    --device cuda:0 \
    --no-terminations

  这个会启动 mjlab env + Viser，但 zero agent 不会真的跟上 reference。要看 ckpt
  tracking，需要用：

  --agent trained --checkpoint-file /path/to/model.pt --motion-file /path/to/
  motion.npz

  所以你现在如果只是想检查 npz 动作长什么样，用第一个 replay_motion_npz_viser.py。
