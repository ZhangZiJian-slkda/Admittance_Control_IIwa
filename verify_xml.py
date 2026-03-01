"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-01 22:02:41
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-01 22:02:43
FilePath: /Admittance_Control_IIwa/verify_xml.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
#!/usr/bin/env python3
"""
验证XML文件是否能正确加载的脚本
"""
import os
import sys

# 检查mujoco是否安装
try:
    import mujoco
except ImportError:
    print("错误: 未安装mujoco, 请先安装: pip install mujoco")
    sys.exit(1)

# 项目路径
project_dir = "/home/zhang/Admittance_Control_IIwa/kuka_iiwa_14"
scene_xml = os.path.join(project_dir, "scene.xml")

print("="*60)
print("KUKA IIWA with Table - XML验证脚本")
print("="*60)

# 检查所有必要的文件是否存在
required_files = [
    "scene.xml",
    "iiwa14.xml",
    "table_assets.xml",
    "table_default.xml",
    "table_body.xml"
]

print("\n检查必要的文件...")
all_files_exist = True
for filename in required_files:
    filepath = os.path.join(project_dir, filename)
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {filename}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n错误: 某些必要的文件缺失!")
    sys.exit(1)

# 尝试加载XML模型
print("\n尝试加载scene.xml...")
try:
    model = mujoco.MjModel.from_xml_path(scene_xml)
    print("✓ 成功加载scene.xml!")
    
    # 打印模型信息
    print("\nMuJoCo模型信息:")
    print(f"  - 自由度数: {model.nq}")
    print(f"  - 速度维度: {model.nv}")
    print(f"  - body数: {model.nbody}")
    print(f"  - geom数: {model.ngeom}")
    print(f"  - 材料数: {model.nmat}")
    
    # 列出所有body名称
    print("\nBody列表:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"  - {body_name}")
    
    print("\n✓ XML验证成功!")
    
except Exception as e:
    print(f"✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*60)
