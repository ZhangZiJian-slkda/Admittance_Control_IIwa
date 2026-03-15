"""
Description: MuJoCo simulation wrapper for the KUKA iiwa14 admittance task
Author: Zhang-sklda 845603757@qq.com
"""
import os

import mujoco
import mujoco.viewer
import numpy as np


XML_PATH = os.path.join(os.path.dirname(__file__), "kuka_iiwa_14")


class IIwaSim:
    def __init__(self, render=True, dt=0.001, xml_path=None):
        if xml_path is None:
            xml_path = os.path.join(XML_PATH, "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kin_data = mujoco.MjData(self.model)
        self.dt = dt
        self.model.opt.timestep = dt
        self.model.opt.gravity[2] = -9.81

        self.render = render
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -45
            self.viewer.cam.azimuth = 90
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        self.step_count = 0
        self.nv = self.model.nv
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_initial_positions = np.array(
            [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0]
        )
        self.force_bias_local = np.zeros(3)

        self.tcp_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        self.force_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, "force_sensor_site"
        )
        self.force_sensor_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor"
        )
        self.tip_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "tip_ball")
        self.tool_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "tool_rod")

        self.reset()
        self.calibrate_force_sensor()
        print("[IIwaSim] Initialized")

    def _require_id(self, obj_type, name):
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Required MuJoCo object not found: {name}")
        return obj_id

    def _forward_data(self, data, q, dq=None):
        data.qpos[:7] = np.asarray(q, dtype=float).reshape(7)
        if dq is None:
            data.qvel[:7] = 0.0
        else:
            data.qvel[:7] = np.asarray(dq, dtype=float).reshape(7)
        data.ctrl[:7] = data.qpos[:7]
        mujoco.mj_forward(self.model, data)
        return data

    def _site_pose(self, data, site_id):
        pose = np.eye(4)
        pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
        pose[:3, 3] = data.site_xpos[site_id]
        return pose

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.joint_initial_positions.copy()
        self.data.qvel[:7] = np.zeros(7)
        self.data.ctrl[:7] = self.joint_initial_positions.copy()
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
            if self.render:
                self.viewer.sync()

        self._forward_data(self.kin_data, self.data.qpos[:7], self.data.qvel[:7])

    def get_state(self):
        return self.data.qpos[:7].copy(), self.data.qvel[:7].copy()

    def get_joint_acceleration(self):
        return self.data.qacc[:7].copy()

    def get_pose(self, q=None):
        if q is None:
            return self._site_pose(self.data, self.tcp_site_id)
        data = self._forward_data(self.kin_data, q)
        return self._site_pose(data, self.tcp_site_id)

    def forward_kinematics(self, q):
        return self.get_pose(q)

    def get_jacobian(self, q=None):
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        data = self.data if q is None else self._forward_data(self.kin_data, q)
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.tcp_site_id)
        return np.vstack((jacp[:, :7], jacr[:, :7]))

    def get_bias_forces(self, q=None, dq=None):
        data = self.data if q is None and dq is None else self._forward_data(
            self.kin_data,
            self.get_state()[0] if q is None else q,
            self.get_state()[1] if dq is None else dq,
        )
        return data.qfrc_bias[:7].copy()

    def get_gravity(self, q=None):
        data = self.data if q is None else self._forward_data(self.kin_data, q, np.zeros(7))
        return data.qfrc_bias[:7].copy()

    def get_ee_force_local(self, subtract_bias=True):
        adr = self.model.sensor_adr[self.force_sensor_id]
        dim = self.model.sensor_dim[self.force_sensor_id]
        force = self.data.sensordata[adr : adr + dim].copy()[:3]
        if subtract_bias:
            force -= self.force_bias_local
        return force

    def get_sensor_force_world(self, subtract_bias=True):
        force_local = self.get_ee_force_local(subtract_bias=subtract_bias)
        rotation = self.data.site_xmat[self.force_site_id].reshape(3, 3)
        return rotation @ force_local

    def get_ee_force_world(self, subtract_bias=True):
        # MuJoCo force sensors report the interaction transmitted through the site body.
        # For admittance control we want the force applied on the tool by the environment.
        return -self.get_sensor_force_world(subtract_bias=subtract_bias)

    def get_ee_force_torque(self, subtract_bias=True):
        force_world = self.get_ee_force_world(subtract_bias=subtract_bias)
        torque_world = np.zeros(3)
        return force_world, torque_world

    def get_applied_body_force(self):
        return self.data.xfrc_applied[self.tip_body_id, :3].copy()

    def calibrate_force_sensor(self, samples=200):
        if samples <= 0:
            self.force_bias_local[:] = 0.0
            return self.force_bias_local.copy()

        adr = self.model.sensor_adr[self.force_sensor_id]
        dim = self.model.sensor_dim[self.force_sensor_id]
        samples_buffer = []

        for _ in range(samples):
            mujoco.mj_step(self.model, self.data)
            if self.render:
                self.viewer.sync()
            samples_buffer.append(self.data.sensordata[adr : adr + dim].copy()[:3])

        self.step_count += samples
        self.force_bias_local = np.mean(samples_buffer, axis=0)
        return self.force_bias_local.copy()

    def step(self):
        self.step_count += 1
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()

    def send_joint_position(self, q_cmd):
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(7)
        q_cmd = np.clip(
            q_cmd,
            self.model.actuator_ctrlrange[:7, 0],
            self.model.actuator_ctrlrange[:7, 1],
        )
        self.data.ctrl[:7] = q_cmd
        self.step()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
