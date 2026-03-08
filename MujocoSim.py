"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-07 22:32:10
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-08 23:24:02
FilePath: /Admittance_Control_IIwa/MujocoSim.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np
import os
import pinocchio as pin
from pinocchio import RobotWrapper
from scipy.spatial.transform import Rotation as R

XML_PATH = os.path.join(os.path.dirname(__file__), 'kuka_iiwa_14')

class IIwaSim:
    def __init__(self, render = True,dt = 0.001,xml_path=None):
        # Load the MuJoCo model
        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(
                os.path.join(XML_PATH, "scene.xml")
            )
        # self.simulated = True
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        self.model.opt.timestep = self.dt
        self.model.opt.gravity[2] = -9.81
        self.step_count = 0
        self.joint_names = [f'joint{i}' for i in range(1, 8)]
        self.joint_initial_positions = np.array([0.0, -0.785398163, 0.0, -1.57, 0.0, 1.57079632679, 0.785398163397])

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -45
            self.viewer.cam.azimuth = 90
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25,0.824])
        else:
            self.render = False

        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        self.nv = self.model.nv
        self.jacobian_position = np.zeros((3,self.nv))
        self.jacobian_rotation = np.zeros((3,self.nv))

        self.M = np.zeros((self.nv,self.nv))
        self.actuator_tau = np.zeros(7)
        self.tau_ff = np.zeros(7)
        self.dq_des = np.zeros(7)

        urdf = os.path.join(XML_PATH, "iiwa14.urdf")
        model = pin.buildModelFromUrdf(urdf)
        self.pin_robot = RobotWrapper(model)

        # 这里后续改成你URDF里真实的末端frame名字
        self.ee_frame_name = "iiwa_link_tcp"
        self.ee_frame_id = self.pin_robot.model.getFrameId(self.ee_frame_name)
        
        self.reset()
        print("[IIwaSim] 初始化成功")

    def forward_kinematics(self,q,update = True):
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        forward_kinematics = self.pin_robot.framePlacement(q, self.ee_frame_id, update_kinematics=update)
        return forward_kinematics.homogeneous

    # 获取机器人状态 由MuJoCo提供
    def get_state(self):
        return self.data.qpos[:7].copy(), self.data.qvel[:7].copy()
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.joint_initial_positions.copy()
        self.data.qvel[:7] = np.zeros(7)
        self.data.ctrl[:7] = self.joint_initial_positions.copy()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
            if self.render:
                self.viewer.sync()

        print("reset qpos:", self.data.qpos[:7])
        print("reset ctrl:", self.data.ctrl[:7])

    def get_gravity(self,q):
        return self.pin_robot.gravity(q)
    
    def get_jacobian(self,q):
        pin.computeJointJacobians(self.pin_robot.model, self.pin_robot.data, q)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

        J_temp = pin.computeFrameJacobian(
            self.pin_robot.model,
            self.pin_robot.data,
            q,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = np.zeros((6, 7))
        J[0:3, :] = J_temp[3:6, :7]   # linear
        J[3:6, :] = J_temp[0:3, :7]   # angular
        return J
    
    def get_pose(self,q):
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        T = self.pin_robot.framePlacement(q, self.ee_frame_id)
        return T.homogeneous
    
    def get_ee_force_torque(self):
        """获取末端执行器的力和力矩传感器数据"""
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor"
        )
        if sensor_id < 0:
            return np.zeros(3)
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        wrench = self.data.sensordata[adr:adr+dim].copy()
        return wrench[:3]   # 这里只取力
    
    def step(self):
        """Execute one simulation step."""
        self.step_count += 1
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
    
    def close(self):
        if self.render:
            self.viewer.close() 

    def send_joint_position(self,q_cmd):
        q_cmd = np.asarray(q_cmd).reshape(7)
        self.data.ctrl[:7] = q_cmd
        self.step()

    def get_ee_force_world(self):
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor"
        )
        if sensor_id < 0:
            return np.zeros(3)

        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        f_local = self.data.sensordata[adr:adr+dim].copy()[:3]  # 取力部分

        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "force_sensor_site"
        )
        R_ws = self.data.site_xmat[site_id].reshape(3, 3)

        f_world = R_ws @ f_local
        return f_world
    
