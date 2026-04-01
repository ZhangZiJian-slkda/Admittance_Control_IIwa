# Admittance Control for KUKA iiwa14 in MuJoCo

这是一个基于 MuJoCo 的 KUKA iiwa14 导纳控制示例项目，主要用于验证机械臂在外力作用下的末端柔顺响应。项目当前提供了：

- `MujocoSim.py`：iiwa14 仿真封装，提供状态读取、位姿/雅可比计算、力传感器读取和位置命令接口
- `admittance.py`：基于笛卡尔空间的导纳控制主程序
- `diagnosis.py`：用于检查模型、雅可比、末端姿态和外力读数的诊断脚本
- `kuka_iiwa_14/`：MuJoCo 模型、场景 XML、URDF 和网格资源
- `Test/`：一些早期测试脚本，包含 torque 控制与 XML/场景验证代码

## Project Structure

```text
Admittance_Control_IIwa/
├── MujocoSim.py
├── admittance.py
├── diagnosis.py
├── requirement.txt
├── readme.md
├── kuka_iiwa_14/
│   ├── scene.xml
│   ├── iiwa14.xml
│   ├── iiwa14.urdf
│   └── assets/
└── Test/
    ├── MujocoSim_test.py
    ├── admittance_test.py
    ├── visualize_scene.py
    └── verify_xml.py
```

## Environment

- Python 3.10+
- Ubuntu/Linux recommended
- OpenGL GUI environment required for MuJoCo viewer

如果你在远程服务器或无图形界面环境运行，请将 `render=False`，否则可视化窗口可能无法启动。

## Installation

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

## Dependencies

主程序依赖：

- `mujoco`
- `numpy`

测试脚本额外依赖：

- `scipy`
- `pin`

说明：

- `pin` 是 `pinocchio` 在 PyPI 上常见的安装包名
- 如果你只运行 `MujocoSim.py`、`admittance.py`、`diagnosis.py`，通常不需要 `pinocchio`

## Quick Start

### 1. 运行导纳控制

```bash
python admittance.py
```

运行后会启动 MuJoCo 可视化窗口。你可以在仿真中对末端施加外部交互，观察末端位置随外力产生柔顺偏移。

### 2. 运行诊断脚本

```bash
python diagnosis.py
```

该脚本会输出：

- 当前关节位置与速度
- TCP 位姿
- Jacobian 尺寸与数值
- 力传感器零偏与世界坐标系下的外力

### 3. 验证场景 XML

```bash
python Test/verify_xml.py
```

### 4. 单独查看场景

```bash
python Test/visualize_scene.py
```

## Control Logic

`admittance.py` 当前实现的是笛卡尔空间导纳控制外环加阻尼伪逆运动学跟踪：

- 根据末端测得外力计算期望笛卡尔速度和位置偏移
- 通过 Jacobian 伪逆把末端速度指令映射到关节速度
- 用 nullspace 项将机械臂拉回初始姿态附近
- 最终通过 MuJoCo 的位置控制接口发送关节目标

这是一个偏研究和实验性质的实现，控制参数需要结合场景接触刚度、时间步长和模型质量继续调参。

## Notes

- 当前主程序使用位置控制，不是力矩控制
- `Test/` 下脚本包含一些历史实验代码，与主流程不完全一致
- `visualize_scene.py` 中仍存在绝对路径写法，如果你更换仓库位置，建议改为相对路径
- MuJoCo 图形界面依赖本地图形环境；WSL、SSH 或 Docker 下可能需要额外配置显示转发

## Recommended Next Improvements

- 增加 `requirements-dev.txt`，区分运行依赖与实验依赖
- 为 `Test/` 脚本统一相对路径
- 增加参数配置文件，避免控制参数硬编码
- 补充导纳控制原理图和实验结果截图

## License

当前仓库未单独声明顶层许可证。如需开源发布，建议补充 `LICENSE` 文件。
