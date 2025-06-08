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

import matplotlib.pyplot as plt
import munch
import numpy as np
#from crazyflow.constants import MASS
from inv_rl.control.quadrotor.attitude_mpc import create_integrator
from mpcc.planners.minsnap_traj.planner_minsnap_sym import PolynomialPlanner
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as Rot
from lsy_drone_racing.control import Controller

## Use my own mass - crazyflow mass is 0.027??
MASS = 0.035

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplineTracker:
    def __init__(self, cs_x_lin, cs_y_lin, cs_z_lin):
        self.cs_x_lin = cs_x_lin
        self.cs_y_lin = cs_y_lin
        self.cs_z_lin = cs_z_lin

    def spline_position(self, t):
        return np.array([self.cs_x_lin(t), self.cs_y_lin(t), self.cs_z_lin(t)])

    def distance_to_spline(self, t, current_position):
        spline_pos = self.spline_position(t)
        d = np.linalg.norm(spline_pos.squeeze() - current_position)
        return d

    def refine_theta(self, t_init, current_position, delta=0.1, tol=1e-4, max_iter=25):
        """Refine theta by minimizing distance to spline.

        :param t_init: Initial guess for parameter t
        :param current_position: np.array([x, y, z])
        :param delta: Search interval half-width
        :param tol: Tolerance for convergence
        :param max_iter: Max number of ternary search steps
        :return: Refined t
        """
        left = max(t_init  - delta, 0.0)
        right = t_init + delta

        for _ in range(max_iter):
            if abs(right - left) < tol:
                break
            t1 = left + (right - left) / 3
            t2 = right - (right - left) / 3

            d1 = self.distance_to_spline(t1, current_position)
            d2 = self.distance_to_spline(t2, current_position)

            if d1 < d2:
                right = t2
            else:
                left = t1

        return (left + right) / 2

    # def refine_theta(self, theta, pos):
    #    if self.distance_to_spline(theta+0.01, pos) > self.distance_to_spline(theta-0.01, pos):
    #        return theta -0.15
    #    else:
    #        return theta +0.15


class AttitudeController(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        # super().__init__(obs, info, config)
        self.freq = config.env.freq
        self.dt = 1 / self.freq
        self.drone_mass = MASS
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._tick = 0


        self._id = info["id"]

        goals = obs["gates_pos"][self._id]
        waypoints = [obs["pos"][self._id]] + goals
        approx_path_length = np.sum(np.linalg.norm(np.diff(waypoints)))

        planner_config = munch.Munch(
            {
                "gate_vel_norm": 0.2,
                "spline_type": "linear",
                "constant_offset": 10.0,
                "num_splines_per_traj_meter": 100.0,
            }
        )

        # planner = PolynomialPlanner(obs, info,planner_config)
        # pos, vel = planner.plan_simple_mujoco(
        #    obs["pos"][self._id],
        #    obs["gates_pos"][self._id],
        #    R.from_quat(obs["gates_quat"][self._id]).as_euler("xyz", degrees=False),
        #    sample_points=approx_path_length / 0.3 * self.freq
        # )

        start_pos = obs["pos"][self._id]
        gates_pos = obs["gates_pos"][self._id]
        gates_quat = obs["gates_quat"][self._id]
        gate_direction = -R.from_quat(gates_quat[-1]).as_matrix()[1, :] * 10.0
        gates_pos = np.concat([gates_pos, gates_pos[[-1]] + gate_direction])
        gates_quat = np.concat([gates_quat, gates_quat[[-1]]])
        gates_rpy = np.zeros((len(gates_quat), 3))

        for i in range(len(gates_quat)):
            # Convert to Euler angles in XYZ order
            q = gates_quat[i, :]
            rot = Rot.from_quat(q)
            gates_rpy[i, :] = rot.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        planner = PolynomialPlanner(obs, info, planner_config)  # self.CTRL_FREQ)
        (
            self.cs_x,
            self.cs_y,
            self.cs_z,
            self.f,
            self.theta_max,
            self.cs_x_lin,
            self.cs_y_lin,
            self.cs_z_lin,
            self.gate_thetas,
            self.lengths,
        ) = planner.plan(start_pos, gates_pos, gates_rpy)

        #time_scaling = info.get("PID_time_scaling", 1.0)
        time_scaling = 10.0
        no_samples = int(approx_path_length * self.freq * time_scaling)
        self.ref = np.zeros((3, no_samples))
        for i, t in enumerate(np.linspace(0, self.theta_max, no_samples)):
            # Didnt find any better way to get casadi DM Values back into numpy - TODO!
            self.ref[:, i] = np.array(
                [self.cs_x_lin(t), self.cs_y_lin(t), self.cs_z_lin(t)]
            ).squeeze()

        self.tracker = SplineTracker(self.cs_x_lin, self.cs_y_lin, self.cs_z_lin)

        # current_position = np.array([x, y, z])  # your measured current position
        # t_init = 5.0  # initial estimate
        theta_guess = 0.01
        refined_t = self.tracker.refine_theta(0.01, obs["pos"][self._id])

        self.x_des = self.ref[0, :]
        self.y_des = self.ref[1, :]
        self.z_des = self.ref[2, :]

        self.ctrl_info = {}
        self.ctrl_info["trajectory"] = self.ref.T
        self.ctrl_info["horizon"] = np.array([])
        self.ctrl_info["opp_prediction"] = np.array([])

        # For data
        self.X = []
        self.U = []

        # Necessary for mpcc state approx.
        self.last_action = np.zeros(4)
        self.last_f_collective = 0.35

        self.last_theta = 0.01
        self.theta = 0.01
        self.v_theta = 0.01
        self.last_v_theta = 0.01
        self.acados_integrator = create_integrator(self.dt * 10, 10)

        self._finished = False

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
        # if i == len(self.x_des) - 1:  # Maximum duration reached
        #self._finished = obs["target_gate"][self._id] == -1
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
        action = np.array([thrust_desired, *euler_desired])

        if self._finished:  # Prevent us from logging data after race finish
            return action, self.ctrl_info

        # Get rpy
        q = obs["quat"][self._id]
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        self.theta = self.tracker.refine_theta(self.theta, obs["pos"][self._id])
        self.f_collective = (
            10.0 * (thrust_desired - self.last_f_collective) * self.dt + self.last_f_collective
        )

        df_cmd, dr_cmd, dp_cmd, _ = (action - self.last_action) / self.dt
        self.v_theta = (self.theta - self.last_theta) / self.dt
        dv_theta = (self.v_theta - self.last_v_theta) / self.dt

        # Create "would be" mpcc state
        mpcc_state = np.concatenate(
            (
                obs["pos"][self._id],
                obs["vel"][self._id],
                rpy,
                np.array([self.theta]),
                np.array([self.last_f_collective]),  # , self.last_f_cmd]), # TODO!
                self.last_action,  # self.last_frpy_cmd,
                np.array([self.v_theta]),
            )
        )
        #print(f"theta: {self.theta}")

        self.X += [mpcc_state]

        mpc_action = np.array([df_cmd, dr_cmd, dp_cmd, dv_theta])
        self.U += [mpc_action]

        self.last_action = action
        self.last_theta = self.theta
        self.last_v_theta = self.v_theta
        self.last_f_collective = self.f_collective
        # return np.concatenate([[thrust_desired], euler_desired], dtype=np.float32), self.ctrl_info
        return action, self.ctrl_info

    def debug_plot_mppc_states(self):
        for i in range(len(self.X[0])):
            plt.figure()
            plt.plot(self.X[:, i])
        plt.show()

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

        # Return Data
        return np.array(self.X), np.array(self.U)

    def episode_reset(self):
        return {}
