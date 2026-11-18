# ELF3 Climbing Tracker

独立的 Isaac Lab 接触条件跟踪训练原型：先学习参考动作与接触先验，再评估任务奖励调整。保留“左夹爪先翻转、再接近并建立前臂支撑”的训练约束。**目前尚未学会完整翻转或攀爬，不是完整 ACP 复现。**

代码与数据分开：本仓库保存代码、配置、测试和必要的 PPO 依赖快照；机器人资产和动作在私有 Release `runtime-v1` 的独立数据包中。无需拷贝整个 TienKung-Lab、GMR 或 retarget_tools。不修改 Hiking，也没有真机执行入口。

## 环境准备与短程诊断

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

### 3. 启动有上限的夹爪翻转诊断训练

```bash
python -u run.py --config configs/flip_4090.toml --mode train \
  --headless --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/flip_prior_001"
```

当前配置为 **128 环境、最多 200 次 PPO 迭代**，每 20 次独立评估并保存；连续 3 次检查退化，或当前阶段评估和采样完成率均低于 20%，自动停止。`flip_4090.toml` 保留历史文件名，本机与 4090 都可使用；4090 显存/速度尚未实测。它是诊断配置，**本轮仍未通过持续站稳验收，不建议直接改成数千轮正式训练**。

输出目录必须是 `runtime/runs/` 内的新目录。`best.pt` 是已评估权重中排序最好的一个，**并不代表合格，也可能是第 0 轮初始化**；需同时检查 `best_iteration` 和对应评估。`latest.pt` / `checkpoint.pt` 对应最后完成评估的权重。若捕获到 Ctrl+C，额外保存未评估的 `interrupted.pt`，不会用上一次评估结果标注中断权重；强制杀进程仍可能无法完成保存。

观察训练：

```bash
tensorboard --logdir "$ELF3_RUNTIME_ROOT/runs" --port 6006
```

主要看 `Curriculum/initial_phase` 是否逐步从 0.9 降至 0，及实际完成率，不以奖励上涨判断学会。训练前及每 20 轮分别记录从头、末段和当前课程阶段的确定性评估；固定场景中的同样初始状态不构成独立随机鲁棒性试验。

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

## 循环观看训练后的策略

```bash
python -u run.py --mode play --segment flip --num-envs 1 \
  --checkpoint "$ELF3_RUNTIME_ROOT/runs/你的训练目录/ppo/model_800.pt" \
  --view rear-left --play-speed 1.0 --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/view_flip_001"
```

不要加 `--headless`。从动作开头反复执行，结束后自动 reset；关闭窗口或 Ctrl+C 退出。`--play-speed 0.5` 半速观看，`--view left/front/right` 改初始视角；也可以在窗口内操作相机。`--play-episodes 2` 只执行两次。播放使用学习策略和物理仿真，不更新权重；慢放不改变物理时间步。`flip` 仅包含左夹爪翻转及站立平衡，不会接近或爬上高台。

## 模块与运行产物

| 路径 | 职责 |
|---|---|
| `src/elf3_climbing/reference.py` | 源动作重采样、接触事件目标与完整性检查 |
| `model.py` / `gripper.py` / `robot_cfg.py` | 最小 FK 模型构建、夹爪包络、原机器人增益快照 |
| `environment.py` | Isaac Lab 场景、观测、动作、奖励与 reset |
| `protocol.py` | 夹爪前置条件、盒体碰撞间隔、CSI 课程 |
| `runner.py` / `settings.py` | 固定 PPO 接口与 TOML 配置 |
| `training.py` / `training_control.py` | 分块训练、评估晋级、失败停止和已评估权重保存 |
| `initialization.py` | 一次性重力负载偏置估计；尚未证明可持续平衡 |
| `evaluation.py` / `calibration.py` / `smoke.py` | 策略评估、FK 校准、可选接口诊断 |
| `cli.py` / `launch.py` / `run.py` | 启动编排和子进程结果检查 |
| `configs/flip_4090.toml` | 4090 起步配置，命令行可覆盖 |
| `scripts/build_runtime.py` / `preflight.py` | 不覆盖源文件的数据打包与迁移检查 |
| `tests/` | 协议、配置、数据及迁移回归测试 |

各次运行保存 `request.json`、`ppo_config.json`、`model_mapping.json`、`fk_calibration.json`、代码/资产哈希。训练另存 `training_contract.json`、`evaluation_before.json`、`evaluation_after.json`、`training_result.json`、`checkpoint.pt`、`best.pt`、`latest.pt`、`evaluation_history.jsonl`、各轮 `evaluation_XXXXX.json`、`initialization.json`、`ppo/` 日志及定期权重。

`completion.json` 只表示程序执行完成，任务是否成功看评估。出现 `error.json` 或缺失完成标记时，顶层启动器返回失败，避免 Isaac 退出码掩盖异常。

## 当前训练约定

- **Actor 632 维 / Critic 697 维，29 动作，200 Hz 物理 / 50 Hz 控制。** Actor 使用本体历史、地形高度与有效掩码、8 槽接触命令和夹爪任务条件；参考全身关节轨迹只用于奖励与 critic，不作为 actor 输入或逐帧动作基线。
- 左 `wrist_y` 目标约 −1.309 rad；误差 ≤0.07 rad、速度 ≤0.25 rad/s、包络分离间隔 ≥1 cm，保持 10 控制步。完成后继续保持角度误差 ≤0.18 rad，禁止超过 1 mm 的包络重叠。翻转前足部向前移动超过 3 cm、或提前建立前臂支撑将终止回合。允许躯干做平衡调整。
- 上述是**仿真训练中的条件与终止规则**，不是已部署硬件 FSM 的安全保证。SAT 间隔符号用于判别盒体重叠，不是精确欧氏距离。
- CSI 从参考进度 0.9 开始。最近 128 次**同一阶段**回合完成率 ≥80%，且单独评估完成率 ≥90%，才前移 0.025；只在 PPO 块结束后晋级。晚阶段的接近/支撑初始化记录 `assumed_prior_flip`，不能当作从头成功的证据。
- 片段完成额外要求：末帧目标姿态、支撑、夹爪门控和速度跟踪条件连续满足 **50 控制步（1 秒）**。平移速度偏差 <0.25 m/s、转动速度偏差 <0.5 rad/s；这些是**完成判定中的实验阈值，不是全程限速或稳定性的充分证明**。例如参考前进 0.7 m/s，实际同样前进 0.7 m/s，速度偏差为零，不会因超过 0.25 m/s 被终止。
- 移动过程中速度奖励跟踪参考运动，允许推进及重心调整。当前短片段的末帧会被固定保持，因此保持阶段目标速度为零；不再沿用重采样边界中残留的非零速度。未来与 Hiking 连续切换或完整攀爬需要相应的移动终点/接触转换条件，不能直接照搬这个固定终点验收。
- 探索标准差固定：腕部 0.005 rad，其余默认 0.01 rad（下肢/腰部可单独配置）。关闭熵项；PPO 每轮采样 48 步、更新 2 遍、clip 0.1、固定学习率 1e-4。这些是已试用的诊断参数，尚不能保证收敛。
- flip 的输出层以末态姿态加静态重力负载补偿初始化一次。使用 MuJoCo 的双脚接触雅可比、单边力及摩擦约束，按实际关节 Kp 换算目标偏置；不在 rollout 中加入参考动作或外力，不修改原增益。腕目标在限位处，补偿会被限位截断，残余力矩写入日志。**这只是初值估计，实际持续平衡仍未通过，不能当作已学会的策略**；`--no-pose-init` 可关闭。其他片段仍用固定末态偏置。
- 动作是名义关节姿态上的策略偏移，经过限位及目标变化率限制。原增益不变，力矩/速度上限取原配置和 URDF 的较小值。
- 可选择 `flip`、`stand`、`prepare`；prepare 表示翻转、接近到前臂支撑，是尚未训练通过的实验片段。未开放完整登膝、收腿与 1 m 攀爬验收。

设计参考 [ACP 原论文](https://arxiv.org/html/2610.31608v1) 的接触条件跟踪与 CSI 思路；本原型尚无学习式接触规划器、联合训练或真实雷达接入。

## 实测状态与限制

2026-11-18 的设计修正发现了两项问题：精密腕部保持被探索噪声破坏；旧验收只要求短暂到位，可能在机器人正在倾倒时提前结束。旧 `model_800.pt` 的 **16/16 是旧条件下的结果，不能继续作为稳定站立通过的证据**。增加躯干速度检查后，对同一旧权重独立复测为 **0/16**。

本轮对照与试训结果（目录均相对 `ELF3_RUNTIME_ROOT/runs/`）：

| 实验 | 实测结果 | 解释 |
|---|---|---|
| `design_exploration_20261118` | 固定姿态、旧验收下，无噪声 128/128；标准差 0.08/腕 0.03 时 0/128；0.01/腕 0.005 时 122/128 | 证实噪声会破坏短时保持；**不能证明持续平衡**。当前脚本按新验收运行，不会复现旧通过率 |
| `design_v3_trial_01` | 末段采样完成率约 96%，下一阶段归零，第 80 轮自动停止 | 小噪声解决了一部分精度问题，未解决全身平衡 |
| `design_v6_trial_03` | 静态负载偏置 + 0.2 秒保持；120 轮后末段 128/128，从头 0/128 | 仍是短暂到位，不能算稳定站立 |
| `design_v7_trial_01` | 1 秒保持；第 60 轮从头与末段均 **0/128**，自动停止 | **FAIL**：持续平衡未学会，课程未晋级；不扩展到接近/攀爬 |

最后的速度语义修正（运动时跟踪参考速度、末帧保持时目标速度归零）另做了小规模接口运行，详见 `validation.json`；它不是收敛实验。每轮完成率、非预期接触、夹爪碰撞等失败原因有独立记录，不再以总奖励曲线代替验收。失败原因可重叠，相加不等于回合数。

**当前结论：训练流程和验收漏洞已修正，学习到持续平衡的效果仍是 FAIL。** 下一步应先定位参考起始站姿在当前碰撞与驱动配置下的失稳过程，并建立能持续平衡的短段先验，再扩大翻转课程；不要在现有失败权重上直接开展数千轮训练、prepare 或风格奖励微调。静态负载偏置本身没有解决这一问题。

当前地形来自解析盒体，不是真实 MID-360 数据。接触标签来自参考几何；前臂/近肘点共用刚体受力读数，不能证明整段前臂承重或无滑移。夹爪质量为用户提供的 **0.5 kg**；质心、惯量仍是包络近似。膝点仍是 link 原点标记；完整 v7 必要力筛查未通过。完整攀爬、1 m 高台、随机扰动验收、真机部署均未通过或未运行。

原机器人资产、Hiking、既有动作和旧权重未修改。复测结果及保护文件检查见 `validation.json`。

## 夹爪质量与惯量

`src/elf3_climbing/gripper_dynamics.json` 保存用户给定的 `mass_kg: 0.5`，随 Python 包一起发布。尺寸及安装方向仍取 runtime 的 `gripper_envelope.json`；无需修改或重新下载 runtime-v1。

两套仿真均把夹爪刚性连接到左 `l_wrist_z_link`：原末端刚体约 0.10656086 kg，加夹爪后约 0.60656086 kg。使用平行轴定理合并质心和完整惯量，碰撞几何不再次贡献质量。仅修改运行时模型，源 URDF/USD、动作和旧权重不变。

**只有 0.5 kg 质量来自用户。** 当前质心取包络几何中心（腕坐标约 `[0.132, 0, 0]` m），惯量按 174×135×70.5 mm 的均匀包络长方体估算，含几何留量；没有额外假设转接件质量。收到实测质心、惯量或转接件质量后应替换近似。

每次启动的 `fk_calibration.json` 会检查 PhysX/MuJoCo 实际加载的末端质量、质心和惯量，并核对 MuJoCo 总质量增量为 0.5 kg。`model_mapping.json` 与训练合同记录质量配置，代码哈希也包含质量 JSON。旧权重允许加载作评估或热启动，但其无夹爪质量的训练结果不能直接沿用；新训练自动包含 0.5 kg，不需新增命令参数。

质量更新验收：27 项测试通过；两引擎质量/质心/惯量检查通过，原模型总质量约 37.87069 kg，加入夹爪后约 38.37069 kg。旧第 800 次迭代权重曾在新质量下按旧条件评估 16/16；这一历史值仅证明当时短时到位，已被本轮更严格的持续平衡复测否定为充分验收依据。历史记录保留在 `validation.json` 的 `gripper_mass_update`。
