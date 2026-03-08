"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-07 22:31:23
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-08 23:23:58
FilePath: /Admittance_Control_IIwa/admittance.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
import numpy as np
import time
from MujocoSim import IIwaSim

my_robot = IIwaSim(render=True)
dt = 0.001
steps = 50000

# 笛卡尔导纳参数
M_adm = np.diag([3.0, 3.0, 3.0])
D_adm = np.diag([60.0, 60.0, 60.0])
K_adm = np.diag([40.0, 40.0, 40.0])

joint, d_joint = my_robot.get_state()
T = my_robot.get_pose(joint)
x_ref = T[:3, 3].copy()

# 导纳状态
delta_x = np.zeros(3)
delta_dx = np.zeros(3)

# 初始关节命令
q_cmd = joint.copy()


for i in range(steps):
    joint, d_joint = my_robot.get_state()
    T = my_robot.get_pose(joint)
    x = T[:3, 3]

    J = my_robot.get_jacobian(joint)
    Jv = J[:3, :]

    dx = Jv @ d_joint

    # 真实传感器力
    F_ext = my_robot.get_ee_force_world()

    # 小力死区，避免噪声抖动
    if np.linalg.norm(F_ext) < 0.8:
        F_ext[:] = 0.0

    # 导纳方程：M * ddx + D * dx + K * x = F
    delta_ddx = np.linalg.solve(
        M_adm,
        F_ext - D_adm @ delta_dx - K_adm @ delta_x
    )

    delta_ddx = np.clip(delta_ddx, -3.0, 3.0)

    delta_dx += delta_ddx * dt
    delta_dx = np.clip(delta_dx, -0.15, 0.15)

    delta_x += delta_dx * dt
    delta_x = np.clip(delta_x, -0.08, 0.08)

    x_des = x_ref + delta_x
    dx_des = delta_dx

    # 笛卡尔位置 + 速度误差反馈
    Kx = np.diag([6.0, 6.0, 6.0])
    v_des = dx_des + Kx @ (x_des - x)

    # resolved-rate inverse kinematics
    dq_cmd = np.linalg.pinv(Jv) @ v_des
    dq_cmd = np.clip(dq_cmd, -0.4, 0.4)

    # q_cmd = q_cmd + dq_cmd * dt
    q_cmd = joint + dq_cmd * dt
    # 可选：限制每步位置变化
    # q_cmd = np.clip(q_cmd, -3.0, 3.0)

    my_robot.send_joint_position(q_cmd)

    if i % 1000 == 0:
        print(f"Step {i}:")
        print("  q      =", joint)
        print("  x      =", x)
        print("  F_ext  =", F_ext)
        print("  delta_x=", delta_x)
        print("  x_des  =", x_des)

    time.sleep(dt)

my_robot.close()