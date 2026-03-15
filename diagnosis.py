"""
Quick diagnostics for the iiwa14 MuJoCo admittance setup.
"""
import numpy as np

from MujocoSim import IIwaSim


def main():
    robot = IIwaSim(render=False, dt=0.001)
    try:
        q, dq = robot.get_state()
        pose = robot.get_pose(q)
        jacobian = robot.get_jacobian(q)
        sensor_force_world = robot.get_sensor_force_world()
        force_world = robot.get_ee_force_world()

        print("joint q:", np.array2string(q, precision=4))
        print("joint dq:", np.array2string(dq, precision=4))
        print("tcp position:", np.array2string(pose[:3, 3], precision=4))
        print("tcp rotation:")
        print(np.array2string(pose[:3, :3], precision=4))
        print("jacobian shape:", jacobian.shape)
        print("linear jacobian:")
        print(np.array2string(jacobian[:3], precision=4))
        print("angular jacobian:")
        print(np.array2string(jacobian[3:], precision=4))
        print("force bias local:", np.array2string(robot.force_bias_local, precision=4))
        print("sensor force world:", np.array2string(sensor_force_world, precision=4))
        print("external force world:", np.array2string(force_world, precision=4))
        print("applied body force:", np.array2string(robot.get_applied_body_force(), precision=4))
    finally:
        robot.close()


if __name__ == "__main__":
    main()
