# ELF3 高台攀爬工作区

| 工作内容 | 入口 |
|---|---|
| 训练、复测、准备环境 | [climbing_tracker 工程说明](README.md) |
| 查看单策略训练框架与 16 项环境奖励 | [0.70 米单台接触策略](docs/contact_training.md) |
| 找到此前 MuJoCo 的 **v4 动作轨迹**，检查或播放原件 | [MuJoCo v4 源轨迹及预览](docs/experiments.md#mujoco-v4) |
| 查看官方控制器、重定向工具 | [官方 ELF3 仿真](https://github.com/bxirobotics/bxi_controller_ros2) · [动作工具](retarget_tools/README.md) |

按本机布局，训练仓库位于 `elf3-climbing-repos/climbing_tracker/`，动作工具随仓保存在 `retarget_tools/`。同级 `bxi_controller_ros2/` 提供官方仿真资产，同级 `output/` 保存示范样本、初始权重和评估产物。运行包使用 `ELF3_RUNTIME_ROOT` 指定位置。
