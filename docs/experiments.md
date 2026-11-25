# 历史实验与复现入口

本页命令从 `climbing_tracker/` 运行。官方 MuJoCo 实验用 31 关节模型在 1 m 台前停靠和支撑，Isaac 分段实验用 29 自由度模型在 0.70 m 单台复现地面走近、送膝交接或台上恢复。v4 源示范的台高约为 0.68 m。

当前单策略训练的输入、奖励和验收见[接触策略训练说明](contact_training.md)。站立、翻腕和官方前臂实验的权重校验值及报告路径见[历史验证索引](../validation.json)。

## 官方台前接近与前臂支撑

此历史入口使用同级 `bxi_controller_ros2/` 的官方资产，以及本机添加的 `tools/elf3_approach.py` 和 `tools/elf3_forearm.py`。复现该实验时需同时准备这两个扩展脚本。

```bash
conda activate isaac_sim_env
cd ~/elf3-climbing-repos/bxi_controller_ros2
python tools/elf3_approach.py --mode forearm --view
```

`--mode approach` 单独观看翻腕、行走与停靠。官方 `model_normal.onnx` 控制步态，左腕采用独立翻转目标。双脚停稳后，全身控制器将两侧前臂放到台面。本机演示使用已知台位、实际尺寸的夹爪包络和 0 kg 附加质量。正向试验中，前臂近端和远端四个位置同时受力约 2.26 秒，末秒夹爪长轴与竖直方向最大夹角约 10.65°。独立记录覆盖 −5°、−3°、−2°、0°、+2°、+3°、+5° 七个初始朝向。

## 保留的策略和结果

下表路径相对于 `$ELF3_RUNTIME_ROOT/runs`。两组 Isaac Lab 独立评估均使用固定场景，各 32/32 个回合完成连续 5 秒保持。

| 用途 | 权重或录像 | 验收记录 |
|---|---|---|
| 站立，左腕保持原位 | `balance_zero_payload_adaptive_01/best.pt` | `balance_zero_payload_eval_01/evaluation.json` |
| 站立翻腕，参考末态初始化后进行腕部 PPO | `flip_wrist_stage_01/best.pt` | `flip_wrist_stage_eval_01/evaluation.json` |
| 官方平地行走、台前停靠 | `bxi_walk_approach_3pfZkBdx/official_approach.mp4` | 同目录 `official_approach.json` 及各朝向的停靠记录 |
| 官方双前臂上台 | `official_forearm_6tEFAncz/forearm.mp4` | 同目录七份 `yaw_*.json` |

`balance_zero_payload_01/best.pt` 是成功站立策略实际使用的第 180 轮初始化权重。在其自身的站立验收中成功率为零，保留它是为了重现权重来源。

### 复测或观看站立翻腕

```bash
conda activate isaac_sim_env
cd ~/elf3-climbing-repos/climbing_tracker
export ELF3_RUNTIME_ROOT="$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1"
export PYTHONPATH="$PWD/src:$PWD/third_party/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
python -u run.py --config configs/flip_from_balance.toml --mode evaluate \
  --checkpoint "$ELF3_RUNTIME_ROOT/runs/flip_wrist_stage_01/best.pt" \
  --num-envs 32 --device cuda:0 --headless \
  --output "$ELF3_RUNTIME_ROOT/runs/flip_eval_$(date +%Y%m%d_%H%M%S)"
```

交互观看时用 `--mode play --num-envs 1 --view rear-left` 替换评估选项，移除 `--headless` 并指定新的输出目录。站立复测使用 `configs/balance.toml` 和表中的站立权重。训练结果看对应片段的 `evaluation.json`，其中包含保持时间和成功回合数。

### 历史行走实验

`walk_from_flip.toml` 记录了从站立翻腕策略派生腰腿动作的实验，默认 1024 环境、2000 轮、固定学习率 0.0001。验收记录要求走完两段速度指令、双脚交替落地并停稳，首次有效落地仅登记侧别。固定学习率实验到第 1800 轮仍为零次有效交替落脚，37 次评估的全程成功率均为零。下面的 MuJoCo v4 实验从官方步态停稳后的交接状态开始。原实验的奖励、计步和停止条件分别见 [walk.py](../src/elf3_climbing/walk.py) 与 [walk_env.py](../src/elf3_climbing/walk_env.py)。

<a id="mujoco-v4"></a>

## MuJoCo v4 动作轨迹

**v4 原件是 `elf3_full_elbow_climb_demo_v4_candidate.npz`**。29 个关节、424 帧、30 Hz，首末帧相隔 14.10 秒，接近地面到台顶的净高度约 0.68 m。模型按前臂、膝、双脚的顺序到达台面。几何检查报告记录了关节限位、穿透与接触位置，动力学状态为 `NOT RUN`。

| 查找对象 | 存放位置 |
|---|---|
| v4 源轨迹、预览与检查 | 数据目录 `$ELF3_DATA_ROOT/elf3/` 下，`elf3_full_elbow_climb_demo_v4_candidate` 对应三份文件：轨迹 `.npz`、视频 `.mp4`、检查报告 `.checks.json`。源 NPZ 的 SHA-256 为 `c3ee7256134a51a77ca8966dff3dc05afb27b4807b40a42b8710ac78030f80ad`。 |
| 由 v4 转换的 Isaac 参考包 | `$ELF3_RUNTIME_ROOT/runs/platform_contacts_20261120_070/isaac_v4_070_surface_reference/reference.npz`，适配 0.70 m 台高，706 帧、50 Hz。同目录的 `manifest.json` 记录源文件和参考包的 SHA-256。 |
| 另一段从地面接近到台面起身的连续动力学录像 | 本地工作区的 `ELF3_完整上台与缓起身.mp4` 对应 `$ELF3_RUNTIME_ROOT/runs/mount_070_ground_to_steady_rise_20261123/trajectory.npz`，仿真时长 21.165 秒。 |

在训练仓库根目录执行以下命令可核对源轨迹并在 MuJoCo 中逐帧回放。显式指定文件名会选中 v4，工具默认选择 v7。

```bash
motion=elf3_full_elbow_climb_demo_v4_candidate
python retarget_tools/elf3_motion.py check --motion "$motion"
python retarget_tools/elf3_motion.py view --motion "$motion" --once
```

### 用 v4 训练肘撑上台

在本仓库根目录用 `isaac_sim_env` 运行 `scripts/platform_train.py`。这个入口使用同级 `bxi_controller_ros2` 的官方 MuJoCo 模型、夹爪包络和 `model_normal.onnx`。每个训练回合从步态停稳、双前臂支撑通过力检的存档开始。已知台高为 1 m，夹爪附加质量为 0 kg。安装命令 `python -m pip install -e '.[platform]'` 可补齐 PPO 入口依赖。

```bash
runs="${ELF3_RUNTIME_ROOT:-$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1}/runs"
for yaw in -3 0 3 5; do
  python scripts/platform_train.py collect --yaw "$yaw" --output "$runs/ascent_handoffs/yaw_$yaw.npz"
done
python scripts/platform_train.py train \
  --handoffs "$runs"/ascent_handoffs/yaw_{-3,0,3}.npz \
  --steps 32768 --output "$runs/ascent_train"
python scripts/platform_train.py evaluate \
  --handoffs "$runs/ascent_handoffs/yaw_5.npz" \
  --checkpoint "$runs/ascent_train/best.zip" --output "$runs/ascent_eval"
```

`--motion` 默认读取本机的 `elf3_full_elbow_climb_demo_v4_candidate.npz`，并核对相邻的 `.checks.json`。训练从这份 0.68 m 示范提取双前臂、右膝、左膝、双脚的台面接触目标，按关节名迁移姿态变化，用官方模型对齐下蹲时的足底与肘部位置。预蹲阶段通过仿真调整关节偏移，模仿初始化之后进行 PPO。训练按接触阶段和实测预蹲等指标保存训练朝向表现最好的权重 `best.zip`，`policy.zip` 是末轮权重。当前阶段使用示范的接触顺序与已知台面，送膝期间有由机器人关节施加的前臂支撑辅助。策略输入含下一处接触目标。奖励和阶段切换使用 MuJoCo 测得的接触力。夹爪或手腕触台不能作为支撑。完整登台的计分要求依次完成预蹲、双脚蹬离地、右膝及左膝承重、双脚上台和站稳保持。已知位姿的接触命令尚待学习式规划器替换。

`training.json` 包含输入来源及训练前后指标，`evaluation.json` 逐朝向记录最高阶段、终止原因和整段成功标志。示例中的训练和评估输出使用新目录。继续训练可加 `--resume /path/to/policy.zip` 并指定新的输出目录。

## 0.70 米单台阶的分段对照实验

这个实验在走近台沿后切换到冻结的第 2300 轮送膝策略，右膝承重后再切到台上恢复控制。第 2300 轮从原扶台起点独立评估时，1024 回合中有 879 次右膝在台面持续承重 0.20 秒。按该权重原有的台面支撑规则复测第 10 轮地面策略，32 回合中有 11 次走到可以切换的位置，送膝成功数为 0。该入口用于复现分段实验。原送膝评估保存在 `$ELF3_RUNTIME_ROOT/runs/inspect_stopped_knee2300_eval_20261122_2215/evaluation.json`。

以下命令复现分段训练，使用 1024 个 Isaac 环境、3000 轮。脚本会打印新的输出目录，每 100 轮保存一次从地面直立开始的完整评估及环境 0 的逐步姿态记录。

```bash
cd ~/elf3-climbing-repos/climbing_tracker
bash scripts/train_ground_mount.sh
```

原地翻腕时，控制器通过踝关节纠正身体漂移。走近台沿时，使用同级 `bxi_controller_ros2` 中官方步态网络的 29 个关节目标，PPO 可以根据实测状态修正动作。环境使用 ONNX 读取已有的权重，依赖版本见 `pyproject.toml` 的 `ground-mount` 项。

每次完整评估都从地面开始，由接触力和机器人姿态决定何时交给送膝策略，再从实际落膝状态继续起身。训练采样也包含行走途中和台上恢复的起点，便于两段同时学习。查看 `history.jsonl` 中的 `ground_mean_peak_flip_drift_m`（原地翻腕期间躯干的最大水平漂移，再对各环境求均值）、`ground_with_alternating_steps`（出现实际换脚落地的回合数）、`ground_walk_loaded_slip_m_s`（行走时承重足的切向滑速）、`ground_braced`（双侧上肢按源策略的接触规则在台面承重、通过交接的回合数）及 `standing_successes`（从地面走到台上，朝向路线并双脚站稳一秒的回合数）。对第 2300 轮的 `tabletop` 模式，单侧支撑力包括前臂壳与腕部在台面顶面上的垂向载荷，侧面或棱边的违规腕部接触仍会阻止交接。

## 从送膝权重连续训练 0.70 米登台

`scripts/train_mount.py` 从原训练目录读取送膝权重、观测归一化参数和参考文件。一次回合从原送膝起点开始。右膝通过原有承重验收后，控制器接手当时的仿真状态，继续尝试移腿和起身。PPO 学习接手后的关节目标修正量。最终验收要求双脚在台面内承重、机器人脱离下层地面、躯干直立并低速保持 1 秒。脚和膝的上台顺序由动作决定。

新训练使用 `flat-foot` 判据。在这一秒的每个控制步，左右脚的足底法线都须距台面法线不超过 15°。对每只脚，程序读取 PhysX 在台面顶面上的实际接触点，按垂向力计算水平位置的协方差。两个主轴上的标准差须分别达到 2.5 cm 和 0.8 cm，以排除单点和沿一条边承重的姿态。回放数据包括左右足倾角及两个方向上的支撑范围。

读取旧权重时默认恢复其原有的 `legacy` 判据。给 `evaluate` 或 `train` 加 `--stance flat-foot`，即可用原权重按新判据重新评估或继续训练。结果和检查点都记录实际采用的判据，其他资产和设置仍逐项核对。

在训练仓库根目录使用 `isaac_sim_env` 运行以下命令。每次输出需选择新目录。

```bash
runs="${ELF3_RUNTIME_ROOT:-$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1}/runs"
entry="$runs/knee_070_contact_transfer_20261122_204705/checkpoint_02300.pt"

python scripts/train_mount.py evaluate --entry-checkpoint "$entry" --num-envs 8 \
  --headless --device cuda:0 --record --output "$runs/mount_baseline"
python scripts/train_mount.py train --entry-checkpoint "$entry" --num-envs 64 \
  --iterations 40 --eval-interval 10 --headless --device cuda:0 --output "$runs/mount_ppo"
python scripts/train_mount.py evaluate --entry-checkpoint "$entry" --checkpoint "$runs/mount_ppo/best.pt" \
  --num-envs 64 --headless --device cuda:0 --record --output "$runs/mount_ppo_eval"
```

`evaluation.json` 分别统计送膝交接、双膝同时承重、任一脚承重、双脚同时承重和最终站稳的回合数。`--record` 默认将环境 0 的姿态、台面载荷及相对台沿位置逐帧写入 `trace.jsonl`。例如，要重看 128 个并行环境中的第 21 号，保持原评估的 `--seed`、权重、仿真设置、`--num-envs 128`，加 `--record-env 21`。各环境使用不同控制参数时，再加 `--control-parameters-file 原评估目录/controller_parameters_by_env.json`。`play` 可在窗口观看选中的环境。

`--controller` 可选 `standard`（默认）、`knee`（先尝试让双膝承重）和 `crawl`（按接触事件移腿）。`--control-parameters '{"tempo":0.9}'` 可覆盖指定参数。`--support-reward` 开启接触后抬高身体的奖励，三种控制器共用上述站稳标准。输出目录中的 `contract.json` 记录控制器、实际参数与源码校验值。用 `--checkpoint` 续训或评估时会恢复这些设置并核对送膝权重和资产。

长训期间每 10 轮更新 `progress.json`，两次评估之间的训练权重保存为 `progress.pt`。每次完整评估均生成编号的 `checkpoint_XXXXX.pt` 和 `evaluation_XXXXX.json`，并更新 `latest.pt`。按台面支撑和站稳表现选出的权重保存在 `best.pt`。开启录制后，每份评估也会生成对应的 `trace_XXXXX.jsonl`。

恢复控制器读取航向坐标系的 9×7 高度网格、有效标记、台沿位置和接触载荷。头部雷达仿真使用 `head_lidar_frame` 和点云话题 `/simulation/head_lidar/points`。点云回放与当前策略的输入见[接触策略训练说明](contact_training.md)。[恢复观测](../src/elf3_climbing/environment.py) · 雷达配置位于同级 `bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation/sensors/head_lidar.xml`。
