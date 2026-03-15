"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-15 22:55:40
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-15 23:39:47
FilePath: /Admittance/Admittance_Control_IIwa/admittance.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
"""
Description: Cartesian admittance control for the MuJoCo KUKA iiwa14 model
Author: Zhang-sklda 845603757@qq.com
"""
import time

import numpy as np

from MujocoSim import IIwaSim


def damped_pseudo_inverse(jacobian, damping=0.08):
    identity = np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + (damping**2) * identity)


def orientation_error_world(rotation_current, rotation_desired):
    return 0.5 * (
        np.cross(rotation_current[:, 0], rotation_desired[:, 0])
        + np.cross(rotation_current[:, 1], rotation_desired[:, 1])
        + np.cross(rotation_current[:, 2], rotation_desired[:, 2])
    )


def main(render=True, steps=50000):
    my_robot = IIwaSim(render=render, dt=0.001)
    dt = my_robot.dt

    # Position-controlled iiwa14 needs a softer outer loop than the FR3 torque case.
    # Use mass-damper admittance so the robot stays at the displaced position
    # after the external force disappears.
    # m_adm = np.diag([2.5, 2.5, 2.5])
    # d_adm = np.diag([90.0, 90.0, 90.0])
    ############
    m_adm = np.diag([1.0, 1.0, 1.0])
    d_adm = np.diag([50.0, 50.0, 50.0])


    joint, _ = my_robot.get_state()
    pose_ref = my_robot.get_pose(joint)
    r_ref = pose_ref[:3, :3].copy()
    q_home = joint.copy()

    x_des = pose_ref[:3, 3].copy()
    dx_des = np.zeros(3)
    force_filtered = np.zeros(3)
    q_cmd = joint.copy()
    force_active_prev = False

    force_alpha = 0.08
    force_deadband = 1.5
    ik_damping = 0.10
    max_joint_velocity = 0.60
    max_joint_step = 0.01
    max_command_error = 0.25
    max_cartesian_speed = 0.20
    max_cartesian_offset = 0.10

    kp_pos = np.diag([5.5, 5.5, 5.5])
    kd_pos = np.diag([0.35, 0.35, 0.35])
    kp_ori = np.diag([3.0, 3.0, 3.0])
    kd_ori = np.diag([0.12, 0.12, 0.12])
    k_null = 0.20

    try:
        for i in range(steps):
            joint, d_joint = my_robot.get_state()
            pose = my_robot.get_pose(joint)
            rotation = pose[:3, :3]
            x = pose[:3, 3]

            jacobian = my_robot.get_jacobian(joint)
            dx = jacobian[:3, :] @ d_joint
            j_angular = jacobian[3:, :]
            omega = j_angular @ d_joint

            f_meas = my_robot.get_ee_force_world()
            force_filtered = (1.0 - force_alpha) * force_filtered + force_alpha * f_meas
            f_ext = force_filtered.copy()
            if np.linalg.norm(f_ext) < force_deadband:
                f_ext[:] = 0.0
            force_active = np.linalg.norm(f_ext) > 0.0

            if force_active_prev and not force_active:
                x_des = x.copy()
                dx_des[:] = 0.0

            ddx_des = np.linalg.solve(
                m_adm,
                f_ext - d_adm @ dx_des,
            )
            ddx_des = np.clip(ddx_des, -4.0, 4.0)

            dx_des += ddx_des * dt
            dx_des = np.clip(dx_des, -max_cartesian_speed, max_cartesian_speed)

            x_des += dx_des * dt
            x_des = np.clip(
                x_des,
                pose_ref[:3, 3] - max_cartesian_offset,
                pose_ref[:3, 3] + max_cartesian_offset,
            )

            v_linear = dx_des + kp_pos @ (x_des - x) - kd_pos @ dx
            v_linear = np.clip(v_linear, -max_cartesian_speed, max_cartesian_speed)

            e_ori = orientation_error_world(rotation, r_ref)
            v_angular = kp_ori @ e_ori - kd_ori @ omega

            twist_cmd = np.concatenate((v_linear, v_angular))
            jacobian_pinv = damped_pseudo_inverse(jacobian, damping=ik_damping)
            dq_task = jacobian_pinv @ twist_cmd

            nullspace_projector = np.eye(7) - jacobian_pinv @ jacobian
            dq_null = k_null * (q_home - joint)
            dq_cmd = dq_task + nullspace_projector @ dq_null
            dq_cmd = np.clip(dq_cmd, -max_joint_velocity, max_joint_velocity)

            joint_step = np.clip(dq_cmd * dt, -max_joint_step, max_joint_step)
            q_cmd += joint_step
            q_cmd = np.clip(q_cmd, joint - max_command_error, joint + max_command_error)
            my_robot.send_joint_position(q_cmd)
            force_active_prev = force_active

            if i % 1000 == 0:
                print(f"step={i}")
                print("  x        =", x)
                print("  x_des    =", x_des)
                print("  F_ext    =", f_ext)
                print("  dx_des   =", dx_des)
                print("  q_cmd    =", q_cmd)

            if render:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        my_robot.close()


if __name__ == "__main__":
    main()
