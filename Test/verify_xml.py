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
"""验证 XML 文件是否能正确加载。"""

import os
import sys

import mujoco


REQUIRED_FILES = [
    "scene.xml",
    "iiwa14.xml",
    "table_assets.xml",
    "table_default.xml",
    "table_body.xml",
]


def resolve_project_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "kuka_iiwa_14")


def validate_scene_xml(project_dir):
    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not os.path.exists(os.path.join(project_dir, filename))
    ]
    if missing_files:
        return False, None, missing_files

    scene_xml = os.path.join(project_dir, "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_xml)
    return True, model, []


def main():
    project_dir = resolve_project_dir()
    print("=" * 60)
    print("KUKA IIWA with Table - XML 验证脚本")
    print("=" * 60)
    print(f"\n检查目录: {project_dir}")

    ok, model, missing_files = validate_scene_xml(project_dir)
    if not ok:
        print("\n错误: 某些必要文件缺失:")
        for filename in missing_files:
            print(f"  ✗ {filename}")
        sys.exit(1)

    print("\n✓ 成功加载 scene.xml!")
    print("\nMuJoCo 模型信息:")
    print(f"  - 自由度数: {model.nq}")
    print(f"  - 速度维度: {model.nv}")
    print(f"  - Body 数: {model.nbody}")
    print(f"  - Geom 数: {model.ngeom}")
    print(f"  - 材料数: {model.nmat}")
    print("\n✓ XML 验证成功!")
    print("=" * 60)


if __name__ == "__main__":
    main()
