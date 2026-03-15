"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2026-03-08 15:29:17
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2026-03-08 17:01:24
FilePath: /Admittance_Control_IIwa/Test/MujocoSim_test.py
Copyright (c) 2026 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
import os
import pinocchio as pin
from pinocchio import RobotWrapper
from scipy.spatial.transform import Rotation as R

XML_PATH = os.path.join(os.path.dirname(__file__), "kuka_iiwa_14")


class IIwaSim:
    def __init__(self, interface_type="torque", render=True, dt=0.001, xml_path=None):
        assert interface_type in ["torque"], "The interface should be torque"
        self.interface_type = interface_type

        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(
                os.path.join(XML_PATH, "scene.xml")
            )
        self.simulated = True
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        _render_dt = 1/60
        self.render_ds_ratio = max(1, _render_dt // dt)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -45
            self.viewer.cam.azimuth = 90
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        else:
            self.render = False

        self.model.opt.timestep = self.dt
        self.model.opt.gravity[2] = -9.81
        self.step_count = 0

        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_initial_positions = np.array(
            [0.0, 0.0, 0.0, -1.57, 0.0, 0.87, 0.785398163397]
        )
        # self.reset()
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        self.nv = self.model.nv

        self.jacobian_position = np.zeros((3,self.nv))
        self.jacobian_rotation = np.zeros((3,self.nv))
        self.M = np.zeros((self.nv,self.nv))

        self.tau_ff = np.zeros(7)
        self.actuator_tau = np.zeros(7)
        self.dq_des = np.zeros(7)
        # urdf = os.path.join(XML_PATH, "iiwa14.urdf")
        # self.pin_robot = RobotWrapper.BuildFromURDF(urdf, package_dirs=[XML_PATH])

        urdf = os.path.join(XML_PATH, "iiwa14.urdf")
        model = pin.buildModelFromUrdf(urdf)
        self.pin_robot = RobotWrapper(model)

        # 这里后续改成你URDF里真实的末端frame名字
        self.ee_frame_name = "iiwa_link_ee"
        self.ee_frame_id = self.pin_robot.model.getFrameId(self.ee_frame_name)

        self.reset()
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, q, update=True):
        T = self.pin_robot.framePlacement(q, self.ee_frame_id, update_kinematics=update)
        return T.homogeneous
    
    def reset(self):
        self.data.qpos[:7] = self.joint_initial_positions.copy()
        self.data.qvel[:7] = np.zeros(7)
        self.data.ctrl[:7] = np.zeros(7)
        mujoco.mj_forward(self.model, self.data)
        if self.render:
            self.viewer.sync()

    def get_state(self):
        return self.data.qpos[:7].copy(), self.data.qvel[:7].copy()

    def get_joint_acceleration(self):
        return self.data.qacc[:7].copy()

    def get_pose(self, q):
        T = self.pin_robot.framePlacement(q, self.ee_frame_id)
        return T.homogeneous

    def get_gravity(self, q):
        return self.pin_robot.gravity(q)[:7].copy()

    def get_dynamics(self, q, v):
        M = self.pin_robot.mass(q)
        h = self.pin_robot.nle(q, v)
        return M, h

    def get_jacobian(self, q):
        # J = self.pin_robot.computeFrameJacobian(q, self.ee_frame_id)
        # return J[:, :7]
        
        J_temp = self.pin_robot.computeFrameJacobian(q, self.ee_frame_id)
        J = np.zeros((6, 7))
        J[3:6, :] = J_temp[0:3, :7]
        J[0:3, :] = J_temp[3:6, :7]
        return J

    def step(self):
        # self.data.ctrl[:7] = self.tau_ff.copy()
        # self.actuator_tau = self.tau_ff.copy()
        # self.step_count += 1
        # mujoco.mj_step(self.model, self.data)
        # if self.render and (self.step_count % self.render_ds_ratio) == 0:
        #     self.viewer.sync()
        tau = self.tau_ff
        self.actuator_tau = tau
        self.data.ctrl[:7] = tau.squeeze()
        self.step_count += 1
        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_count % self.render_ds_ratio) == 0:
            self.viewer.sync()


    def send_joint_torque(self, torques):
        # torques = np.asarray(torques).reshape(-1)
        # if torques.shape[0] != 7:
        #     raise ValueError("torques must be 7-dimensional")
        # self.tau_ff = torques.copy()
        # self.step()
        self.tau_ff = np.asarray(torques).reshape(7)
        self.latest_command_stamp = time.time()
        self.step()

    def close(self):
        if self.render:
            self.viewer.close()

    def compute_tau(self, q, v, ddq_des):
        ddq_des = np.append(ddq_des,[0.0,0.0])
        M,h = self.get_dynamics(q,v)
        tau = M @ ddq_des + h
        return tau[:7]
