"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.constants import MASS
from inv_rl.attitude_mpc_wrapper import LearningController
from mpcc.control.controller_single import ControllerSingle as MPCC

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_controller_custom import AttitudeController as AttCtrl

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        super().__init__(obs, info, config)
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

        info["id"] = 0
        self.controller_0 = AttCtrl(obs, info, config)
        # self.controller_0 = LearningController(obs, info, config)
        # w = self.controller_0.w
        # scale = info.get("MPCC_weight_scale", 1.0)
        # assert 0.85 <= scale <= 0.95
        # w[6] *= scale
        # self.controller_0.ctrl.update_weights(w)
        info["id"] = 1
        self.controller_1 = MPCC(obs, info, config)

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
        action = np.zeros((1, 2, 4))
        action[0, 0, :], ctrl_info_0 = self.controller_0.compute_control(obs, info)
        action[0, 1, :], ctrl_info_1 = self.controller_1.compute_control(obs, info)
        return action, (ctrl_info_0, ctrl_info_1)

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
        ctrl_finished_0 = self.controller_0.step_callback(
            action, obs, reward, terminated, truncated, info
        )
        ctrl_finished_1 = self.controller_1.step_callback(
            action, obs, reward, terminated, truncated, info
        )
        # Make sure this makes sense
        self._finished = ctrl_finished_0 & ctrl_finished_1
        return self._finished

    def episode_callback(self, **kwargs):
        """Reset the integral error."""
        self._tick = 0
        # This is the learning controller that is used to control the other drone
        X, U = self.controller_0.episode_callback()
        # This is the MPCC
        self.controller_1.episode_callback(X=X, U=U)
        return

    def episode_reset(self):
        persistent_info_0 = self.controller_0.episode_reset()
        persistent_info_1 = self.controller_1.episode_reset()
        return (persistent_info_0, persistent_info_1)
