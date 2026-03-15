"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-08 16:04:08
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-08 16:08:24
FilePath: /Admittance_Control_IIwa/Test/admittance_test.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
import numpy as np
import time
import pinocchio as pin
from MujocoSim_test import IIwaSim

my_robot = IIwaSim(render=True, dt=0.001)

dt = 0.001
steps = 50000

# 导纳参数
# M_adm = np.diag([1.0, 1.0, 1.0])
# D_adm = np.diag([120.0, 120.0, 120.0])
# K_adm = np.diag([80.0, 80.0, 80.0])

M_adm = np.diag([1.0, 1.0, 1.0])
D_adm = np.diag([300.0, 300.0, 300.0])
K_adm = np.diag([2000.0, 2000.0, 2000.0])

# 任务空间跟踪增益（先保守一点）
Kp_task = np.diag([1500.0, 1500.0, 1500.0])
Kd_task = np.diag([300.0, 300.0, 300.0])

# 初始状态
q, dq = my_robot.get_state()
T0 = my_robot.get_pose(q)
x_ref = T0[:3, 3].copy()

# 导纳内部状态：位移偏移 和 偏移速度
delta_x = np.zeros(3)
delta_dx = np.zeros(3)

# MuJoCo末端body名字要和你的xml一致
ee_body_id = my_robot.model.body(b"link7").id   # 如果报错，再换成你的真实末端body名

try:
    for i in range(steps):
        q, dq = my_robot.get_state()
        T = my_robot.get_pose(q)
        x = T[:3, 3]
        R_ee = T[:3, :3]

        # 取Jacobian
        J = my_robot.get_jacobian(q)

        # 当前你的 get_jacobian 返回的是：
        # J[0:3] = linear
        # J[3:6] = angular
        # 如果你后面按这个定义，就不要再来回交换了
        Jv = J[0:3, :]
        dx = Jv @ dq

        # 外力（MuJoCo交互拖拽施加）
        F_ext = my_robot.data.xfrc_applied[ee_body_id, :3].copy()

        # 简单死区，避免微小扰动引发漂移
        if np.linalg.norm(F_ext) < 1.0:
            F_ext[:] = 0.0

        # 导纳外环： M*ddelta_x + D*delta_dx + K*delta_x = F_ext
        delta_ddx = np.linalg.solve(
            M_adm,
            F_ext - D_adm @ delta_dx - K_adm @ delta_x
        )

        # 限幅，防止数值爆炸
        delta_ddx = np.clip(delta_ddx, -5.0, 5.0)

        delta_dx += delta_ddx * dt
        delta_dx = np.clip(delta_dx, -0.3, 0.3)

        delta_x += delta_dx * dt
        delta_x = np.clip(delta_x, -0.10, 0.10)

        # 导纳生成的目标位置和目标速度
        x_des = x_ref + delta_x
        dx_des = delta_dx

        # 内环：任务空间PD跟踪
        pos_error = x_des - x
        vel_error = dx_des - dx
        F_cmd = Kp_task @ pos_error + Kd_task @ vel_error

        # 再限幅一下期望力
        F_cmd = np.clip(F_cmd, -80.0, 80.0)

        tau = Jv.T @ F_cmd + my_robot.get_gravity(q)

        # 关节力矩限幅（非常重要）
        tau = np.clip(tau, -100.0, 100.0)

        my_robot.send_joint_torque(tau)

        time.sleep(dt)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    my_robot.close()
