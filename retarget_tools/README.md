# ELF3 攀爬动作工具

本目录保留当前 v7 生成、回放、检查及原始数据重定向所需的代码。数据目录为 `~/humanoid_datasheet/datasets/elf3_climbing/`。

历史 MuJoCo **v4** 已在 `python elf3_motion.py list` 中标出。原始轨迹、预览、核验信息和指定 v4 的回放命令见 [MuJoCo v4 动作轨迹](../docs/experiments.md#mujoco-v4)。

## 常用命令

```bash
conda activate isaac_sim_env
cd ~/elf3-climbing-repos/climbing_tracker/retarget_tools
python elf3_motion.py paths
python elf3_motion.py list
python elf3_motion.py check
python elf3_motion.py view --once
python elf3_motion.py view --view side --once
```

统一入口默认选择 `elf3_full_elbow_climb_demo_v7_candidate`，支持 `--motion 动作名或NPZ路径`。回放时空格暂停、方向键逐帧。`check` 检查字段、关节映射及首尾帧 FK，不代表接触或动力学验收通过。

导出三视角视频、逐帧图谱或恢复段检查结果，需要指定尚不存在的目录：

```bash
python elf3_motion.py render --output ~/humanoid_datasheet/datasets/elf3_climbing/elf3/validation/my_new_review
python elf3_motion.py audit --work ~/humanoid_datasheet/datasets/elf3_climbing/elf3/validation/my_new_audit
python elf3_motion.py run build_ordered_recovery.py -- --help
```

`audit` 依赖 v6 对照、固定阶段名和原场景，不能用于任意轨迹。退出码 0 表示检查完成，验收状态看输出 JSON。`run` 转发原脚本参数，不额外提供输出保护。保留脚本的原命令仍可使用：

```bash
python view_motion.py --motion elf3_full_elbow_climb_demo_v7_candidate --view rear --once
```

## 当前状态

v7 候选为 478 帧、30 Hz，包含夹爪预调整、前臂上台、有序收腿及协调起身。它是模型约束和优化生成的候选动作，不是未经修改的 GMR 输出。几何检查不等于动力学通过：接触力必要条件筛查仍有失败帧，闭环动力学尚未运行，不能标为训练验收通过，也不是已验证的 1 m 攀爬。

数据目录中的主要证据：

- `elf3/elf3_full_elbow_climb_demo_v7_candidate.npz` 及同名 `.checks.json`。
- `elf3/validation/v7_multiview_review/`：视频和逐帧图谱。
- `elf3/validation/v7_ordered_recovery/final_audit/`：几何与接触力检查。
- `elf3/validation/v7_ordered_recovery/final_provenance.json`：生成来源与检查点。

## 代码分工与依赖

`tool_catalog.json` 列出当前工具；`local_imports` 只记录静态导入。

| 作用 | 主要代码 |
|---|---|
| 统一入口、路径配置 | `elf3_motion.py`、`project_paths.py` |
| 回放、视频、审查 | `view_motion.py`、`inspect_multiview_motion.py`、`render_climbing_demo.py`、`audit_*.py` |
| v7 生成与修正 | `build_ordered_recovery.py`、`refine_ordered_support_keys.py`、`polish_ordered_edge.py` |
| 共享模型与优化辅助 | `plan_knee_entry.py`、`refine_recovery_demo.py`、`gripper_geometry.py`、`bvh.py` |
| 碰撞距离与轨迹导出 | `refine_climb_clearance.py`、`build_full_elbow_demo.py` |
| 数据筛选与重定向 | `retarget.py`、`screen_support.py`、`check_motion.py` |

不能按旧文件名删除的依赖：

```text
build_ordered_recovery → refine_recovery_demo.Recovery
                      → plan_knee_entry.create_context → 模型与夹爪几何

audit_foot_separation → 从 refine_climb_clearance 提取 closest / gjk
refine_climb_clearance → 子进程调用 build_full_elbow_demo 导出
```

`refine_recovery_demo.py` 和 `plan_knee_entry.py` 的共享功能仍被当前流程使用，但其独立主程序属于早期方案，直接运行不等于执行 v7 流程。

## 路径与复现

`project_paths.py` 集中配置路径，可在启动命令前用环境变量覆盖：

| 环境变量 | 默认值（相对于工作区） |
|---|---|
| `ELF3_WORKSPACE_ROOT` | 用户主目录（`$HOME`） |
| `ELF3_DATA_ROOT` | `humanoid_datasheet/datasets/elf3_climbing` |
| `ELF3_TIENKUNG_ROOT` | `TienKung-Lab` |
| `ELF3_GMR_ROOT` | `GMR` |

模型在 `TienKung-Lab/legged_lab/assets/elf3_lite/`，夹爪包络使用本目录 `gripper_envelope.json`。模型内部引用和历史数据路径未做整体迁移。

场景上下文依赖 `elf3/omniretarget_climb_28_z_scale_0.8.npz`。修改 v7 前阅读 `final_provenance.json`，使用对应 `--reuse-keys`、`--resume-states` 检查点并写入新目录。单独运行默认生成脚本不保证复现已有 v7 的完整生成链及元数据。

## 本地源码归档

早期重定向脚本和当时的工具清单保存在 `../../_archive/retarget_tools_legacy_20261118_113114.tar.gz`。官方模型的 1 m 台前支撑演示与训练基线见[历史实验](../docs/experiments.md)。
