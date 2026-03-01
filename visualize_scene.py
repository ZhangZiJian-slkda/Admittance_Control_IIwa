"""
KUKA IIWA with Table - MuJoCo可视化脚本
"""
import os
import mujoco
import mujoco.viewer
import numpy as np

project_dir = "/home/zhang/Admittance_Control_IIwa/kuka_iiwa_14"
scene_xml = os.path.join(project_dir, "scene.xml")

print("="*60)
print("KUKA IIWA with Vention Table - MuJoCo可视化")
print("="*60)

print(f"\n加载场景文件: {scene_xml}")
model = mujoco.MjModel.from_xml_path(scene_xml)
data = mujoco.MjData(model)

print(f"✓ 模型加载成功!")
print(f"\n模型信息:")
print(f"  - 关节数(DOF): {model.nq}")
print(f"  - Body数: {model.nbody}")
print(f"  - 几何体数: {model.ngeom}")

print(f"\n启动MuJoCo可视化查看器...")
print("提示: 可以使用鼠标与查看器交互:")
print("  - 左键拖动: 旋转视图")
print("  - 右键拖动: 缩放视图")
print("  - 中键拖动: 平移视图")
print("  - Ctrl+右键: 拖动物体")
print("  - 按 'H' 键: 显示帮助信息")
print("  - 按 'Q' 键或关闭窗口: 退出")
print()

# 设置初始姿态
q0 = np.array([0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397])
data.qpos[:7] = q0
mujoco.mj_step(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.5])

    print("✓ 查看器已启动")
    print("关闭窗口或按 Ctrl+C 退出...")

    try:
        while viewer.is_running():
            step_start = data.time
            while (data.time - step_start) < 1.0 / 60.0:
                mujoco.mj_step(model, data)
            viewer.sync()
    except KeyboardInterrupt:
        print("\n✓ 已退出查看器")

print("="*60)
print("仿真结束")
print("="*60)