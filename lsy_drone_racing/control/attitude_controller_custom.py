"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.constants import MASS
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import munch
from lsy_drone_racing.control import Controller
from mpcc.planners.minsnap_traj.planner_minsnap_sym import PolynomialPlanner


if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController:
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        #super().__init__(obs, info, config)
        self.freq = config.env.freq
        self.drone_mass = MASS
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._tick = 0

        self._finished = False

        self._id = info["id"]

        goals = obs["gates_pos"][self._id]
        waypoints = [obs["pos"][self._id]] + goals
        approx_path_length = np.sum(np.linalg.norm(np.diff(waypoints)))

        planner_config = munch.Munch({"gate_vel_norm":0.15,
                                      "spline_type": "linear",
                                      "constant_offset": 10.0})
        planner = PolynomialPlanner(obs, info,planner_config)
        pos, vel = planner.plan_simple_mujoco(
            obs["pos"][self._id], 
            obs["gates_pos"][self._id],
            R.from_quat(obs["gates_quat"][self._id]).as_euler("xyz", degrees=False),
            sample_points=approx_path_length / 0.3 * self.freq
        )
        print(f"pos: {pos}")

        self.x_des = pos[0, :]
        print(f"shape xde: {np.shape(self.x_des)}")
        self.y_des = pos[1, :]
        self.z_des = pos[2, :]
        print(f"att controller instanciated with id {self._id}")

        self.ctrl_info = {}
        self.ctrl_info["trajectory"] = pos.T
        self.ctrl_info["horizon"] = np.array([])
        self.ctrl_info["opp_prediction"] = np.array([])


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        i = min(self._tick, len(self.x_des) - 1)
        if i == len(self.x_des) - 1:  # Maximum duration reached
            self._finished = True

        des_pos = np.array([self.x_des[i], self.y_des[i], self.z_des[i]])
        des_vel = np.zeros(3)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"][self._id]
        vel_error = des_vel - obs["vel"][self._id]

        # Update integral error
        self.i_error += pos_error * (1 / self.freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"][self._id]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)
        thrust_desired = max(thrust_desired, 0.3 * self.drone_mass * self.g)
        thrust_desired = min(thrust_desired, 1.8 * self.drone_mass * self.g)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
        thrust_desired, euler_desired
        return np.concatenate([[thrust_desired], euler_desired], dtype=np.float32), self.ctrl_info

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self.i_error[:] = 0
        self._tick = 0
        print(f"Episode Callback AttitudeController.")
    
    def episode_reset(self):
        return {}

