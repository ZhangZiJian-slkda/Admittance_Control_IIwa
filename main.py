"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-01 22:05:59
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-01 22:06:00
FilePath: /Admittance_Control_IIwa/main.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
'''
Author: Zhang-sklda 845603757@qq.com
Date: 2025-11-22 14:47:29
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2025-11-23 23:38:29
FilePath: /kuka_iiwa_mujoco_tutorial/IKPY_Test/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import mujoco.viewer
import time

def main():
    # Load and display the MuJoCo model
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    data.ctrl[:7] = [0, 0, 0, -1.57, 0, 0, 0]  # Set initial joint positions

    ee_body = model.body('link7').id
    ee_pos = data.xpos[ee_body]
    ee_quat = data.xquat[ee_body]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            # print("末端位置 (m):", ee_pos)
            # print("末端姿态 (quat):", ee_quat)
            time.sleep(0.002)

if __name__ == "__main__":
    main()