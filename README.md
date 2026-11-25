# ELF3 高台攀爬

[工作区首页与目录](WORKSPACE.md)

当前 Isaac Lab 主线训练一个 29 自由度策略，让机器人从地面走近 0.70 m 单台、双臂扶台、蹬地，把右膝送上台顶并连续承重 0.50 秒。要查看奖励、启动 1024 个无头环境、复测检查点或导出录像，请打开[接触策略训练说明](docs/contact_training.md)。

| 工作内容 | 入口与数据 |
|---|---|
| 训练框架、16 项环境奖励、示范学习、评估与回放 | [单策略训练指南](docs/contact_training.md) · [训练脚本](scripts/train_contact_tracker.sh) |
| **此前的 MuJoCo v4 动作轨迹** | [v4 源文件、预览和回放命令](docs/experiments.md#mujoco-v4)。原动作 424 帧、30 Hz，当前训练采用它派生的 0.70 m、50 Hz 参考包。 |
| 官方 1 m 前臂支撑、站立翻腕、分段送膝和上台起身 | [历史实验与复现](docs/experiments.md) · [2026-11-20 验证索引](validation.json) |
| 动作重定向与几何检查、官方控制器 | [动作工具](retarget_tools/README.md) · [官方仿真](https://github.com/bxirobotics/bxi_controller_ros2) |

## 代码位置

| 职责 | 实现 |
|---|---|
| 加载策略、PPO、示范更新、选取检查点 | [contact_runner.py](src/elf3_climbing/contact_runner.py) |
| 仿真步进、重置、累计回合奖励 | [contact_tracker_env.py](src/elf3_climbing/contact_tracker_env.py) |
| 换步、搭台、送膝与验收 | [contact_tracker_task.py](src/elf3_climbing/contact_tracker_task.py) · [接触测量](src/elf3_climbing/isaac_knee_entry.py) |
| 预测落点并根据实际接触评分 | [接触点策略](src/elf3_climbing/contact_point_actor.py) · [接触计划奖励](src/elf3_climbing/contact_plan_reward.py) |
| 采集示范、加载地面与送膝教师、计算模仿损失 | [采集脚本](scripts/collect_contact_examples.py) · [教师加载](src/elf3_climbing/mount_teacher.py) · [示范学习](src/elf3_climbing/contact_learning.py) |
| 复测记录、头部雷达点云回放 | [轨迹复盘](scripts/review_contact_attempt.py) · [点云回放](scripts/replay_head_lidar.py) |

## 环境与数据

工程使用 Python 3.11、Isaac Sim 5.1.0.0、Isaac Lab 0.54.4、PyTorch 2.7.0+cu128 和 MuJoCo 3.11.0。先准备对应的 `isaac_sim_env`，在训练仓库根目录安装项目及随仓提供的 RSL-RL。[依赖版本和文件校验值](DEPENDENCIES.json)

```bash
conda activate isaac_sim_env
cd ~/elf3-climbing-repos/climbing_tracker
python -m pip install -e third_party/rsl_rl --no-deps
python -m pip install -e '.[test]'
export ELF3_RUNTIME_ROOT="${ELF3_RUNTIME_ROOT:-$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1}"
```

基础模型、夹爪配置和参考动作位于 `$ELF3_RUNTIME_ROOT`。新机器可从私有 Release 获取基础包并校验后解压。复现当前接触策略还需迁移对应的 v4 派生参考包、示范数据和初始化权重，位置见[训练数据与启动命令](docs/contact_training.md#启动复测与回放)。

```bash
data_dir="$(dirname "$ELF3_RUNTIME_ROOT")"
mkdir -p "$data_dir/transfers"
gh release download runtime-v1 --repo hywxj/elf3-climbing-tracker \
  --pattern 'elf3-runtime-v1.tar.gz*' --dir "$data_dir/transfers"
(cd "$data_dir/transfers" && sha256sum -c elf3-runtime-v1.tar.gz.sha256) && \
  test ! -e "$data_dir/runtime_v1" && tar -xzf "$data_dir/transfers/elf3-runtime-v1.tar.gz" -C "$data_dir"
```

## 验证工程

`preflight.py` 核验运行包和 PPO 依赖，`--require-cuda` 同时检查训练设备。pytest 会收集接触判定、奖励、起点、模型加载、独立评估和雷达输入等回归测试。策略的实际成功率由另外运行的地面评估给出。

```bash
python scripts/preflight.py --require-cuda
python -m pytest -q tests
```
