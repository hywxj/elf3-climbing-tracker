# ELF3 Climbing Tracker

独立的 Isaac Lab 接触条件跟踪训练原型。**平地站立及参考初始化的夹爪翻转已有固定场景验证；当前推进无视觉平地行走，行走能力尚未验收通过。** 附加夹爪质量为 0 kg，保留安装尺寸和碰撞包络。后续再做策略切换、接近和前臂支撑，最后做负载适应。目前不是完整 ACP 复现，也未完成攀爬。

代码与数据分开：本仓库保存代码、配置、测试和必要的 PPO 依赖快照；机器人资产和动作在私有 Release `runtime-v1` 的独立数据包中。原 tracker 无需拷贝整个 TienKung-Lab、GMR 或 retarget_tools；新增 `run_walk.py` 目前依赖本地定制 TienKung-Lab 及其已整理的 AMP 数据，不能只下载 runtime-v1 就在另一台机器运行。不修改 Hiking，也没有真机执行入口。

## 无视觉平地行走（2026-11-19）

`run_walk.py` 复用本地 `walk_elf3` 的环境和 AMP/PPO，扩展代码位于本仓库的 `walk_env.py`、`walk_data.py`、`walk_cli.py`，不改共享工程。原 `walk_elf3` 默认有障碍地形，本入口显式改为平面；`elf3_continuous_flat` 虽关闭相机，却向演员附加解析地形信息，因此没有直接采用它。

- Actor：角速度、重力方向、速度命令、29 个关节角偏差/速度/上一动作，96 维 × 10 帧 = 960 维。无深度、雷达、高度图或解析地形。Critic 另外读取真实根线速度与双足接触，共 1010 维。
- 200 Hz 物理、50 Hz 控制，29 个动作按实际关节名称绑定缩放。腿部允许学习；不沿用翻转阶段的 wrist-only 冻结。**旧站立/翻转检查点没有直接迁入，新行走演员从头学习**，两个入口的观测和动作合同不同。
- 从约 −75° 的已翻转腕姿开始，学习前进 0.10～0.30 m/s；25% 的重采样命令为停步，每 2～4 秒换一次命令。不是从未翻转状态完成整段动作，翻转→行走的在线切换仍为 NOT RUN。
- 同一零附加质量的夹爪长方体被附加到每个仿真环境。腕目标为 −1.30898 rad，其他关节没有锁死。软限位使用模型真实范围，避免旧 90% 范围把初始翻转角缩回约 7.5°；没有扩大机械限位。
- 已有 GMR 数据中 `walk_turn` 类别的 214 条合格候选（168 训练、46 验证）复制到本轮输出的 `walk_data/`，校验原 SHA-256，保留原分组划分。楼梯、斜坡、碎石、跑步机和根节点固定动作不纳入本轮。合格表示通过既有数据筛选，不表示动力学行走已通过。
- AMP 同时屏蔽专家与模拟状态的左 `wrist_y` 角度/速度两列（AMP70 的 12、41），避免专家姿态拉回夹爪。保留其他特征；手端特征沿用原肘框架虚拟点，不是夹爪接触点。夹爪保持由独立奖励监督。

任务奖励先乘 0.02 s，再与同样按时间缩放的 AMP 奖励按 `0.7 × task + 0.3 × AMP` 混合；停步命令屏蔽 AMP。任务连续系数：速度 +5（向量误差 ≤0.05 m/s 满分，超出按 0.20 m/s 尺度平滑下降）、偏航速度 +1、翻转保持 +1.5、直立 −3、横滚/俯仰角速度 −0.1、垂向速度 −1、关节加速度 −2.5e−7、动作变化 −0.01、能量 −2e−4、限位 −2、足滑移 −0.5、单脚支撑/腾空时长 +1、停步时关节运动 −0.02。失败系数 −100 也经过时间和混合缩放，每次终止实际贡献 −1.4。基础 AMP 系数为 2。没有“速度误差必须精确为零”的条件。

本机启动一个新的正式实验（不会自动停止在第 60 轮；3000 是实验预算，不保证收敛）：

```bash
cd /home/hamlet/elf3-climbing-repos/climbing_tracker
conda activate isaac_sim_env
export ELF3_RUNTIME_ROOT="$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1"
export TIENKUNG_ROOT="$HOME/TienKung-Lab"
export PYTHONPATH="$PWD/src:$PWD/third_party/rsl_rl"
WALK_RUN="$ELF3_RUNTIME_ROOT/runs/blind_walk_$(date +%Y%m%d_%H%M%S)"
python -u run_walk.py --mode train --headless --num-envs 512 \
  --iterations 3000 --device cuda:0 --output "$WALK_RUN"
```

每 100 轮和正常结束时保存检查点，`latest.pt` 只在正常完成时写入。中断时使用最近的 `model_N.pt`，不存在已验收的 `best.pt`。`--checkpoint` 只接受本行走入口且合同匹配的检查点；不会加载 Hiking 或旧站立模型。运行目录含 `env.yaml`、`agent.yaml`、`contract.json`、数据来源、TensorBoard、结果和完成标记。查看学习曲线时优先看 `Walk/episode_seconds`、`Walk/failure_fraction`、`Walk/velocity_error_m_s` 和 `Walk/wrist_error_rad`，总奖励不能替代评估。

训练后独立检查；保留同一终端的 `$WALK_RUN`，或设为实际运行目录：

```bash
python -u run_walk.py --mode evaluate --headless --num-envs 32 --device cuda:0 \
  --checkpoint "$WALK_RUN/latest.pt" \
  --output "$ELF3_RUNTIME_ROOT/runs/walk_eval_$(date +%Y%m%d_%H%M%S)"
# 可视化：同一命令改为 --mode play --num-envs 1，并去掉 --headless。
```

检查流程为站立 2 s→0.2 m/s 前进 4 s→停稳 4 s。通过要求：全过程没有摔倒/非足接触失败，稳态速度误差 <0.10 m/s、腕角峰值误差 <0.18 rad、最后 2 s 平均速度 <0.08 m/s、前进段位移 >0.4 m；至少 95% 环境满足。发生失败后的自动重置不会抹掉失败记录。输出的全环境速度/位移均值仅用于诊断，包含失败后的重置状态；不能当作连续成功轨迹。此协议也不代表已会定位高台或完成与翻转策略的切换。

短测证据统一见 `validation.json`：16 环境两轮集成、512 环境有限更新与独立评估分开记录。**短测权重尚未证明会走，不作为推荐部署或迁移权重。** 新入口当前依赖本机定制环境；迁移 4090 前需另行整理对应依赖与数据包。

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

### 3. 当前任务：无附加负载的平地站立

```bash
python -u run.py --config configs/balance.toml --mode train \
  --headless --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/balance_$(date +%Y%m%d_%H%M%S)"
```

默认 512 环境、300 轮，每 20 轮评估。`balance` 使用源动作第 0 帧的未翻腕站姿，移除高台，只保留足底所在的地面；不修改参考包。原有 `stand` 是登台后的收尾片段，**不是本任务**。站立任务不使用 CSI、不加载旧翻转权重；左 `wrist_y` 目标固定为 0 rad，通过正常关节执行器保持，其他关节由策略控制。网络维度保持不变，夹爪任务条件改为 `[0, 0]`，表示未翻腕且不要求翻转。

站立奖励保留关节姿态、根位置、线速度、角速度、根姿态、关节速度、双脚位置与支撑，以及动作平滑和力矩惩罚；**翻转进度、翻转到位和前臂姿态奖励全部关闭**。速度奖励仍允许小误差（0.05 m/s、0.1 rad/s 内满分），不要求机器人完全不动。没有翻转门控、丢失翻转、翻转前迈步失败；真实包络碰撞、摔倒和非预期上肢支撑仍会终止回合。

每回合最多 10 秒，目标是连续 5 秒满足位置偏差 <0.15 m、姿态偏差 <0.4 rad、关节均方误差 <0.12 rad²、双脚各 >5 N、线速度 <0.25 m/s、角速度 <0.5 rad/s。PPO 初始学习率 3e-4、按 KL 自适应调节、clip 0.2、5 遍更新。站立任务只在连续两次完整 5 秒评估 ≥90%、而带噪声采样成功率 <80% 时降低腰腿探索；1 秒探测仅用于观察，不再触发降噪或热启动降噪。这是基础动作学习，不做承重能力和真机部署验收。

观看时用 `--mode play --segment balance --num-envs 1 --checkpoint <本次目录>/best.pt --view rear-left --output <新的运行目录>`，不要加 `--headless`。独立评估使用 `--mode evaluate --segment balance --num-envs 32 --checkpoint ... --headless --device cuda:0 --output ...`。是否学会以评估结果为准，不能仅看文件名 `best.pt`。

本机当前结果：`balance_zero_payload_adaptive_01/best.pt` 已在独立进程重新加载，32/32 个固定初始状态回合通过连续 5 秒站立（`balance_zero_payload_eval_01/evaluation.json`）。未翻腕，附加质量 0 kg。54 项自动测试通过。此权重来自先前站立试训第 180 轮，再用当前调度热启动 200 轮；不是当前配置从零训练已验证收敛的证明。带探索噪声的末块训练成功率仍为 0，未做随机扰动或负载验证。详细证据及失败试验保留在 `validation.json`。

本机直接观看已验证的权重：

```bash
PYTHONPATH=src:third_party/rsl_rl python -u run.py --mode play \
  --segment balance --num-envs 1 --device cuda:0 --view rear-left \
  --checkpoint "$ELF3_RUNTIME_ROOT/runs/balance_zero_payload_adaptive_01/best.pt" \
  --output "$ELF3_RUNTIME_ROOT/runs/balance_view_$(date +%Y%m%d_%H%M%S)"
```

### 站稳后的下一阶段：参考初始化的夹爪翻转

已能站立后，使用 `configs/flip_from_balance.toml`，配合
`--checkpoint "$ELF3_RUNTIME_ROOT/runs/balance_zero_payload_adaptive_01/best.pt"`。
此配置为 512 环境、最多 200 轮、每 20 轮评估；平地、附加质量 0 kg、从第 0 帧开始、不启用倒序课程，不接近高台。
加载 balance 权重时会自动初始化此前恒定的夹爪任务输入、腕部历史输入及被屏蔽的腕输出，
腕部输出以参考第 73 帧约 −1.309 rad 初始化，而物理 reset 仍是第 0 帧的 0 rad；目标按已有 1.5 rad/s 限制渐变。运行中不写 qpos、不逐帧注入参考角度。`standing_transfer.json` 明确记录此初始化，训练前成功不能算成 PPO 从零学习的成果。
当前 `wrist_only=true`：只更新左腕输出行和 critic；站立 actor 的隐藏层、其余输出行、actor 归一化参数保持不变。腰腿仍执行已有的实时站立反馈，**不固定身体关节**。腰腿等其他动作的采样标准差固定 0.001 rad，左腕 0.005 rad。本阶段腕输出依赖已有特征，尚不是全身技能联合训练；后续解冻全身需要单独的技能保持约束。
`best.pt` 优先按完整完成率和连续保持选择；未完成时进一步比较存活率和腕角误差，再比较速度，避免优先选中完全不翻腕的站姿。
不要用普通 `--segment flip --checkpoint <balance权重>` 绕过该配置。
历史 v1 只给腕输出 0 rad 初值，在用户运行到约 420 轮时翻转仍几乎没有进展；v2 改为参考末态初始化。单独保留初始化、开放全身更新的试验又出现站立退化，因此不推荐沿用该试验的 latest.pt。最新实验结果及初始化/训练的区别见 `validation.json`。播放此阶段权重也应使用该配置，以保持平地场景一致。

当前复测：`flip_wrist_stage_01` 的 40 轮专项 PPO 后，末块带小幅动作噪声的 1024/1024 个训练回合完成任务，确定性评估 512/512；独立进程加载 `best.pt` 后 32/32 完成翻转并连续站稳 5 秒（`flip_wrist_stage_eval_01`）。只有腕输出的两组参数张量更新，站立反馈及 actor 归一化逐项不变。**参考末态初始化在 PPO 之前已能完成任务**，这些结果不能表述为从零训练学会；随机扰动与全身微调尚未验证。60 项自动测试通过。

```bash
PYTHONPATH=src:third_party/rsl_rl python -u run.py \
  --config configs/flip_from_balance.toml --mode play --num-envs 1 \
  --device cuda:0 --view rear-left \
  --checkpoint "$ELF3_RUNTIME_ROOT/runs/flip_wrist_stage_01/best.pt" \
  --output "$ELF3_RUNTIME_ROOT/runs/flip_view_$(date +%Y%m%d_%H%M%S)"
```

### 历史翻转诊断配置（当前先不运行）

```bash
python -u run.py --config configs/flip_4090.toml --mode train \
  --headless --device cuda:0 \
  --output "$ELF3_RUNTIME_ROOT/runs/flip_prior_001"
```

当前配置为 **128 环境、最多 200 次 PPO 迭代**，每 20 次独立评估并保存；连续 3 次检查已建立的完成能力退化会自动停止；“无任务成功”停止从第 200 轮才开始计数（默认最早第 240 轮），仍需连续 3 次当前阶段评估和采样完成率均低于 20%，且完整评估中的平均最长连续保持时间没有相对已记录进度增加至少 0.1 秒。显式迭代预算始终生效。`flip_4090.toml` 保留历史文件名，本机与 4090 都可使用；4090 显存/速度尚未实测。它是诊断配置，**本轮仍未通过持续站稳验收，不建议直接改成数千轮正式训练**。

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

各次运行保存 `request.json`、`ppo_config.json`、`model_mapping.json`、`fk_calibration.json`、代码/资产哈希。训练另存 `training_contract.json`、`evaluation_before.json`、`evaluation_after.json`、`training_result.json`、`checkpoint.pt`、`best.pt`、`latest.pt`、`evaluation_history.jsonl`（含噪声调度）、各轮 `evaluation_XXXXX.json`、`initialization.json`、`ppo/` 日志及定期权重。

`completion.json` 只表示程序执行完成，任务是否成功看评估。出现 `error.json` 或缺失完成标记时，顶层启动器返回失败，避免 Isaac 退出码掩盖异常。

## 翻转及攀爬片段训练约定（balance 差异见上文）

- **Actor 632 维 / Critic 697 维，29 动作，200 Hz 物理 / 50 Hz 控制。** Actor 使用本体历史、地形高度与有效掩码、8 槽接触命令和夹爪任务条件；参考全身关节轨迹只用于奖励与 critic，不作为 actor 输入或逐帧动作基线。
- 左 `wrist_y` 目标约 −1.309 rad；误差 ≤0.07 rad、速度 ≤0.25 rad/s、包络分离间隔 ≥1 cm，保持 10 控制步。完成后继续保持角度误差 ≤0.18 rad，禁止超过 1 mm 的包络重叠。翻转前足部向前移动超过 3 cm、或提前建立前臂支撑将终止回合。允许躯干做平衡调整。
- 上述是**仿真训练中的条件与终止规则**，不是已部署硬件 FSM 的安全保证。SAT 间隔符号用于判别盒体重叠，不是精确欧氏距离。
- CSI 从参考进度 0.9 开始。最近 128 次**同一阶段**回合完成率 ≥80%，且单独评估完成率 ≥90%，才前移 0.025；只在 PPO 块结束后晋级。晚阶段的接近/支撑初始化记录 `assumed_prior_flip`，不能当作从头成功的证据。
- 片段完成额外要求：末帧目标姿态、支撑、夹爪门控和速度跟踪条件连续满足保持时间。**flip 现在为 250 控制步（5 秒）**，stand/prepare 仍为 50 步（1 秒）。5 秒用于持续支撑的训练验收，不规定未来 FSM 的切换等待时间。平移速度偏差 <0.25 m/s、转动速度偏差 <0.5 rad/s；这些是**完成判定中的实验阈值，不是全程限速或稳定性的充分证明**。例如参考前进 0.7 m/s，实际同样前进 0.7 m/s，速度偏差为零，不会因超过 0.25 m/s 被终止。
- 移动过程中速度奖励跟踪参考运动，允许推进及重心调整。当前短片段的末帧会被固定保持，因此保持阶段目标速度为零；不再沿用重采样边界中残留的非零速度。未来与 Hiking 连续切换或完整攀爬需要相应的移动终点/接触转换条件，不能直接照搬这个固定终点验收。
- 探索按部位和阶段区分：腕部固定 0.005 rad，其他上肢 0.01 rad，腰腿从 0.08 rad 开始。连续两次当前阶段 1 秒探测通过率 ≥90%、而完整保持的采样完成率 <80% 时，腰腿噪声减半至最低 0.01 rad。**短探测只调节噪声，不批准课程晋级**；晋级依然需要完整保持评估 ≥90% 和采样完成率 ≥80%。加载检查点后，如果训练前短探测通过率 ≥90%，默认腰腿噪声不超过 0.02 rad 且不高于保存值；显式 `--balance-noise-std` 可覆盖，便于对照。关闭熵项；PPO 每轮采样 48 步、更新 2 遍、clip 0.1、固定学习率 1e-4。这些是已试用的诊断参数，尚不能保证收敛。
- flip 的输出层以末态姿态加静态重力负载补偿初始化一次。使用 MuJoCo 的双脚接触雅可比、单边力及摩擦约束，按实际关节 Kp 换算目标偏置；不在 rollout 中加入参考动作或外力，不修改原增益。腕目标在限位处，补偿会被限位截断，残余力矩写入日志。**这只是初值估计，实际持续平衡仍未通过，不能当作已学会的策略**；`--no-pose-init` 可关闭。其他片段仍用固定末态偏置。
- 动作是名义关节姿态上的策略偏移，经过限位及目标变化率限制。原增益不变，力矩/速度上限取原配置和 URDF 的较小值。
- 可选择 `balance`、`flip`、`stand`、`prepare`；prepare 表示翻转、接近到前臂支撑，是尚未训练通过的实验片段。未开放完整登膝、收腿与 1 m 攀爬验收。

设计参考 [ACP 原论文](https://arxiv.org/html/2610.31608v1) 的接触条件跟踪与 CSI 思路；本原型尚无学习式接触规划器、联合训练或真实雷达接入。

## 翻转 prior 奖励（v9；balance 关闭三项上肢任务奖励）

速度目标跟踪允许小误差：躯干平移速度误差 ≤0.05 m/s、转动速度误差 ≤0.1 rad/s 时，该项奖励满分，范围内梯度为零；超出后使用 `exp(-max(error-tolerance, 0)^2 / 0.25)` 平滑降低。比较的是速度向量差的模，不是限制前进速度。比如参考 0.70 m/s、实际 0.67 m/s，平移速度奖励仍满分。

这两个值是**奖励满分范围**。末尾保持验收仍要求速度偏差 <0.25 m/s、角速度偏差 <0.5 rad/s，并与姿态、双脚支撑、夹爪到位等条件一起连续满足保持时长（当前 flip 为 5 秒）；不要求误差精确为零，也不把验收阈值用于全程限速。

下面是 `reward_stage=prior` 的实际奖励。每步将连续项相加后乘以控制周期 **0.02 s**；表中正系数为该项乘时间前的上限，负系数乘对应平方误差。`mean` 是对关节或参考激活接触点求平均。

| 连续项 | 系数 | 计算/用途 |
|---|---:|---|
| 关节姿态模仿 | +1.0 | `exp(-mean((q-q_ref)^2)/0.15)` |
| 根节点位置跟踪 | +2.0 | `exp(-||p-p_ref||^2/0.04)` |
| 根节点线速度跟踪 | +2.0 | 0.05 m/s 误差内满分，超出平滑降低 |
| 根节点角速度跟踪 | +0.5 | 0.1 rad/s 误差内满分，超出平滑降低 |
| 根节点朝向跟踪 | +0.5 | `exp(-orientation_error^2/0.25)` |
| 关节速度模仿 | +0.1 | `exp(-mean((qdot-qdot_ref)^2)/10)`；末尾保持目标速度为零 |
| 夹爪翻转进度 | +1.5 | `clamp(1-abs(wrist_error)/1.309, 0, 1)` |
| 夹爪到位稳定 | +0.6 | `exp(-wrist_error^2/0.01-wrist_speed^2/0.0625)` |
| 支撑时夹爪包络朝向 | +0.3 | `exp(-up_angle^2/0.12)`，只在前臂保持阶段启用；flip 阶段为零 |
| 参考接触点位置 | +1.0 | 激活点的 `exp(-position_error^2/0.0025)` 均值 |
| 参考接触实际建立 | +0.3 | 激活点对应刚体对地形受力 >3 N 的比例；不是点级力分布证明 |
| 动作变化惩罚 | −0.02 | `mean((action_t-action_prev)^2)`，抑制命令跳变 |
| 关节力矩惩罚 | −0.00005 | `mean(applied_torque^2)` |

终止事件直接加减，不再乘 0.02：成功奖励 `2 + 10.5 × 剩余回合秒数`；失败终止扣 1；未成功而超时扣 0.5，触发重叠时可叠加。提前迈步、丢失夹爪翻转、夹爪穿入平台、姿态/位置偏差过大、非预期碰撞等属于失败判定，不是另列的多个连续惩罚。正连续项上界仍为 9.8/s，成功奖励保留了提前完成优于拖延的设计。

奖励版本、满分误差范围写入 `training_contract.json`。TensorBoard 新增线/角速度误差及对应奖励率。v9 已通过小误差非零满分、范围外平滑下降等单测及 2 环境 20 步接口检查，**没有在本次修改后执行 PPO 收敛试验**。

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

**v9 历史结论：持续平衡未通过。v10 新增下方的失败定位与分阶段探索验证，不能把这些历史结果当作新版本通过证据。** 下一步应先定位参考起始站姿在当前碰撞与驱动配置下的失稳过程，并建立能持续平衡的短段先验，再扩大翻转课程；不要在现有失败权重上直接开展数千轮训练、prepare 或风格奖励微调。静态负载偏置本身没有解决这一问题。

当前地形来自解析盒体，不是真实 MID-360 数据。接触标签来自参考几何；前臂/近肘点共用刚体受力读数，不能证明整段前臂承重或无滑移。夹爪质量为用户提供的 **0.5 kg**；质心、惯量仍是包络近似。膝点仍是 link 原点标记；完整 v7 必要力筛查未通过。完整攀爬、1 m 高台、随机扰动验收、真机部署均未通过或未运行。

原机器人资产、Hiking、既有动作和旧权重未修改。复测结果及保护文件检查见 `validation.json`。

## 本次失败定位与 v10 修正

用户的 `flip_v9_20261118_175616` 请求 3000 轮，实际因 `no_task_success` 在 60 轮停止；不是 3000 轮训练已经耗尽。1024 环境、约 295 万环境步，从头和末段均未完成。均值奖励 6.51、关节误差较小，不能抵消实际失稳。

`balance_failure_trace_01/balance_trace.json` 的单环境逐步回放显示：双脚仍接触地面，身体持续向前运动，踝部目标却接近常量；约 0.96 s 左腕出现约 462 N 接触力。训练后策略和静态负载偏置均未形成足够的闭环平衡。腕角误差小并不意味着身体稳定。已将“丢失腕角”与“包络碰撞”分开统计，失败判定的并集不变，未放宽碰撞限制。

`balance_model_audit_01/model_audit.json` 检查所有刚体的惯性参数，并在原姿态和两组非零关节扰动下比较两引擎 FK：最大位置差约 0.51 微米，姿态、质量、质心和惯量检查通过。**不包含接触求解器、碰撞几何或驱动动态等价性证明。** 可用 `scripts/diagnose_balance.py --model-audit` 复查；脚本不修改源模型或检查点。

在同一失败权重基础上，将腰腿探索设为 0.08 rad、允许 200 轮的 `balance_exploration_probe_01`，第 100、120 轮在进程内的末段 1 秒保持评估均为 512/512，但采样完成率仍为零、从头仍未通过。它同时增加了探索和可用迭代数，不能单独归因于某个参数。另起进程重新加载后短探测约 58.4%，而 `balance_learned_hold5_01` 的 5 秒测试在约 2.28 s 后终止；因此**短时达标不等于可靠持续站立**。

v10 据此引入分阶段腰腿探索、无成功停止的初始学习窗口、flip 末尾 5 秒保持，以及明确分开的 1 秒探测。每次评估记录 `required_hold_s`、`max_stable_hold_s` 和四个受禁止接触刚体的终端受力；不得拿 `short_probe` 的成功率替代 `current`/`from_start` 的完整验收。速度小误差满分区、奖励系数、机械限位、0.5 kg 夹爪及原关节增益保持原设定。

最后的 `balance_low_noise_hold5_probe_01` 运行 512 环境、200 轮，腰腿噪声由 0.02 降至 0.01：1 秒探测 **511/512**，从头及末段的 5 秒保持均 **0/512**；最佳第 120 轮的末段平均最长连续保持 **2.019 秒**。**持续平衡 FAIL，课程仍为 0.9，暂不建议照旧开展 3000 轮正式训练。** 此后按用户要求改为先做无附加质量、不翻腕的 balance 站立任务（见上文）。Hiking 教师迁移暂未执行，不能假定不同策略的动作和观测能直接拼接。

本轮各组延长保持试训的状态、人工中断原因和最终测量统一保存在 `validation.json`。当前仍以该文件中的完整保持结果判定；实验性调度本身不构成收敛保证。Hiking 原策略只做了只读检查，尚未迁移成站立教师或修改其权重。

## 夹爪质量与惯量

`src/elf3_climbing/gripper_dynamics.json` 当前设为 **`mass_kg: 0.0`**（训练附加质量）；`physical_mass_kg: 0.5` 仅记录用户给定实物质量，不参与当前仿真。尺寸及安装方向仍取 runtime 的 `gripper_envelope.json`；无需修改或重新下载 runtime-v1。此默认值对新启动的 balance/flip/stand/prepare 都生效，已经运行的进程不会自动改变。

当前两套仿真都保留包络碰撞，但跳过夹爪惯性合并；原腕刚体仍约 0.10656086 kg，总质量约 37.87069 kg。源 URDF/USD、动作和旧权重不变。未来将 `mass_kg` 逐步增大时才启用平行轴定理合并，碰撞几何不另加质量。

**只有实物 0.5 kg 质量来自用户。** 恢复负载后可用的近似质心取包络中心（腕坐标约 `[0.132, 0, 0]` m），惯量按 174×135×70.5 mm 均匀包络长方体估算。当前质量为零，不增加这部分质心偏移或惯量。

每次启动的 `fk_calibration.json` 核对实际加载的质量和配置相符，当前增量应为 **0 kg**。`model_mapping.json` 与训练合同记录配置，代码哈希包含质量 JSON。新增 balance 任务拒绝加载旧 flip 检查点；站立从头学习，或明确加载 balance 权重。

质量更新验收：27 项测试通过；两引擎质量/质心/惯量检查通过，原模型总质量约 37.87069 kg，加入夹爪后约 38.37069 kg。旧第 800 次迭代权重曾在新质量下按旧条件评估 16/16；这一历史值仅证明当时短时到位，已被本轮更严格的持续平衡复测否定为充分验收依据。历史记录保留在 `validation.json` 的 `gripper_mass_update`。
