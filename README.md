# ELF3 Climbing Tracker

独立的 Isaac Lab 接触条件跟踪训练原型：先学习参考动作与接触先验，再评估任务奖励调整。保留“左夹爪先翻转、再接近并建立前臂支撑”的训练约束。**目前尚未学会完整翻转或攀爬，不是完整 ACP 复现。**

代码与数据分开：本仓库保存代码、配置、测试和必要的 PPO 依赖快照；机器人资产和动作在私有 Release `runtime-v1` 的独立数据包中。无需拷贝整个 TienKung-Lab、GMR 或 retarget_tools。不修改 Hiking，也没有真机执行入口。

## 4090 上启动

### 1. 准备环境和代码

使用 Linux / Python 3.11、Isaac Sim **5.1.0.0**、Isaac Lab **0.54.4**、PyTorch **2.7.0+cu128**。这是本机已验证的组合，其他版本尚未验证。Isaac Lab 源码提交为 `b4c321024792976150ca55fddb26fa34480d974e`（官方 `isaac-sim/IsaacLab`）；可在独立环境中按该版本的安装说明安装。仓库不会自动覆盖现有 Isaac 环境。

在已安装上述 Isaac 环境的终端执行：

```bash
gh repo clone hywxj/elf3-climbing-tracker
cd elf3-climbing-tracker
conda activate isaac_sim_env  # 换成 4090 机器上对应的环境名
python -m pip install -e third_party/rsl_rl --no-deps
python -m pip install -e .
```

`third_party/rsl_rl` 固定了此前实际使用的本地 2.3.1 版本，保留许可证；不要直接换成最新 RSL-RL。它使用旧式 tuple 观测接口，外层由 `runner.py` 适配。安装步骤不会下载模型，也不会开始训练。

### 2. 下载模型和动作

私有仓库需要 GitHub 账户访问权限。数据目录可以更换，但应在启动 Python 前设置 `ELF3_RUNTIME_ROOT`。

```bash
mkdir -p "$HOME/humanoid_datasheet/datasets/elf3_climbing/transfers"
gh release download runtime-v1 --repo hywxj/elf3-climbing-tracker \
  --pattern 'elf3-runtime-v1.tar.gz*' \
  --dir "$HOME/humanoid_datasheet/datasets/elf3_climbing/transfers"
(
  cd "$HOME/humanoid_datasheet/datasets/elf3_climbing/transfers"
  sha256sum -c elf3-runtime-v1.tar.gz.sha256
)
# 首次解压使用新目录，避免覆盖已有实验。
tar -xzf "$HOME/humanoid_datasheet/datasets/elf3_climbing/transfers/elf3-runtime-v1.tar.gz" \
  -C "$HOME/humanoid_datasheet/datasets/elf3_climbing"
export ELF3_RUNTIME_ROOT="$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1"

python scripts/preflight.py --require-cuda
python -m unittest discover -s tests -v
```

数据包含原 ELF3 URDF/MJCF/USD/网格、左夹爪包络配置、v7 源动作与 50 Hz 参考包。文件逐项 SHA-256 校验；manifest 使用相对路径，换机器后无需编辑。原 v7 为 **478 帧 / 30 Hz**，训练参考为 **796 帧 / 50 Hz**；约 **0.68 m** 高台，不能宣称已覆盖 1 m。

### 3. 启动夹爪翻转先验训练

```bash
python -u run.py --config configs/flip_4090.toml --mode train \
  --headless --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/flip_prior_001"
```

配置为 **128 环境、2000 次 PPO 迭代**，每 100 次保存 checkpoint。128 是供 4090 起步的配置，**尚未在 4090 实测显存或速度**。可以覆盖，例如 `--num-envs 64 --iterations 500`。输出目录必须是 `runtime/runs/` 内的新目录，避免覆盖实验。Ctrl+C 可以停止；已写入的定期 checkpoint 保留，不保证中断当步会额外保存。

观察训练：

```bash
tensorboard --logdir "$ELF3_RUNTIME_ROOT/runs" --port 6006
```

主要看 `Curriculum/initial_phase` 是否逐步从 0.9 降至 0，及实际完成率，不以奖励上涨判断学会。训练结束分别记录从头与末段的确定性评估；固定场景中的同样初始状态不构成独立随机鲁棒性试验。

### 4. 独立评估与权重继续训练

```bash
python -u run.py --mode evaluate --segment flip --num-envs 32 \
  --checkpoint "$ELF3_RUNTIME_ROOT/runs/flip_prior_001/checkpoint.pt" \
  --headless --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/flip_eval_001"
```

`--checkpoint` 也可用于 `--mode train`，加载网络及观测归一化；**它是权重热启动，未实现恢复优化器、CSI 课程和仿真状态的完整断点续训**。跨片段从 flip 切到 prepare 同样需要重新学习。不要把旧试训失败权重当作合格先验。`--reward-stage refine` 必须明确指定 checkpoint，目前仅降低关节模仿权重，不自动转阶段，夹爪约束保持不变。

可选接口检查（不做梯度更新，也不是正式训练的前置验收）：

```bash
python -u run.py --mode smoke --smoke-controller zero --segment flip \
  --num-envs 2 --steps 20 --headless --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/interface_001"
```

这里 zero 表示名义姿态对应的零策略动作，底层仍使用关节位置驱动。`reference-pd` 仅保留为可选诊断，不要求它通过后才能 RL。

## 模块与运行产物

| 路径 | 职责 |
|---|---|
| `src/elf3_climbing/reference.py` | 源动作重采样、接触事件目标与完整性检查 |
| `model.py` / `gripper.py` / `robot_cfg.py` | 最小 FK 模型构建、夹爪包络、原机器人增益快照 |
| `environment.py` | Isaac Lab 场景、观测、动作、奖励与 reset |
| `protocol.py` | 夹爪前置条件、盒体碰撞间隔、CSI 课程 |
| `runner.py` / `settings.py` | 固定 PPO 接口与 TOML 配置 |
| `evaluation.py` / `calibration.py` / `smoke.py` | 策略评估、FK 校准、可选接口诊断 |
| `cli.py` / `launch.py` / `run.py` | 启动编排和子进程结果检查 |
| `configs/flip_4090.toml` | 4090 起步配置，命令行可覆盖 |
| `scripts/build_runtime.py` / `preflight.py` | 不覆盖源文件的数据打包与迁移检查 |
| `tests/` | 协议、配置、数据及迁移回归测试 |

各次运行保存 `request.json`、`ppo_config.json`、`model_mapping.json`、`fk_calibration.json`、代码/资产哈希。训练另存 `training_contract.json`、`evaluation_before.json`、`evaluation_after.json`、`training_result.json`、`checkpoint.pt`、`ppo/` 日志及定期权重。

`completion.json` 只表示程序执行完成，任务是否成功看评估。出现 `error.json` 或缺失完成标记时，顶层启动器返回失败，避免 Isaac 退出码掩盖异常。

## 当前训练约定

- **Actor 632 维 / Critic 697 维，29 动作，200 Hz 物理 / 50 Hz 控制。** Actor 使用本体历史、地形高度与有效掩码、8 槽接触命令和夹爪任务条件；参考全身关节轨迹只用于奖励与 critic，不作为 actor 输入或逐帧动作基线。
- 左 `wrist_y` 目标约 −1.309 rad；误差 ≤0.07 rad、速度 ≤0.25 rad/s、包络分离间隔 ≥1 cm，保持 10 控制步。完成后继续保持角度误差 ≤0.18 rad，禁止超过 1 mm 的包络重叠。翻转前足部向前移动超过 3 cm、或提前建立前臂支撑将终止回合。允许躯干做平衡调整。
- 上述是**仿真训练中的条件与终止规则**，不是已部署硬件 FSM 的安全保证。SAT 间隔符号用于判别盒体重叠，不是精确欧氏距离。
- CSI 从参考进度 0.9 开始，最近 128 次对应阶段回合完成率 ≥80% 时前移 0.1；阈值是 ELF3 实验设定。较晚的接近/支撑初始化会显式记录 `assumed_prior_flip`，不能作为从头翻转成功的证据。
- 默认以片段终态设置网络输出层的固定姿态偏置，帮助末段课程起步；这不是已经学习到的策略，训练前评估会记录它。`--no-pose-init` 可关闭。
- 动作是名义关节姿态上的策略偏移，经过限位及目标变化率限制。原增益不变，力矩/速度上限取原配置和 URDF 的较小值。
- 可选择 `flip`、`stand`、`prepare`；prepare 表示翻转、接近到前臂支撑，是尚未训练通过的实验片段。未开放完整登膝、收腿与 1 m 攀爬验收。

设计参考 [ACP 原论文](https://arxiv.org/html/2610.31608v1) 的接触条件跟踪与 CSI 思路；本原型尚无学习式接触规划器、联合训练或真实雷达接入。

## 实测状态与限制

2026-11-18 历史试训：32 环境、200 次 PPO 迭代、153600 个环境步，actor 参数确实更新；从头完成率为 **0/32**，末段由训练前 **32/32** 降至训练后 **0/32**，课程未晋级。不能视为学会翻转。之后已修正拖到超时比提前完成收益更高的奖励问题，并调整提前接近判定、探索噪声；**修正后未再训练**，等待在 4090 上验证。

本次重构执行离线测试、路径迁移校验、包构建及 2 环境零动作接口检查，**没有新增 PPO 更新**。测量记录见 `validation.json`。原机器人资产、Hiking 和已有参考文件未修改。

当前地形来自解析盒体，不是真实 MID-360 数据。接触标签来自参考几何；前臂/近肘点共用刚体受力读数，不能证明整段前臂承重或无滑移。夹爪仅有碰撞包络，质量惯量未标定。膝点仍是 link 原点标记；完整 v7 必要力筛查未通过。现阶段适合训练接口与短段先验实验，不能据此宣称动力学验收或真机可用。

下一步由 4090 执行 flip 先验实验，依据从头完成率决定是否扩展 prepare；完整接触先验未学会前，不自动进入风格奖励调整或完整攀爬训练。
