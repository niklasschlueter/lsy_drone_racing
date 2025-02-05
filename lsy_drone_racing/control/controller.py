"""Base class for controller implementations.

Your task is to implement your own controller. This class must be the parent class of your
implementation. You have to use the same function signatures as defined by the base class. Apart
from that, you are free to add any additional methods, attributes, or classes to your controller.

As an example, you could load the weights of a neural network in the constructor and use it to
compute the control commands in the :meth:`compute_control <.BaseController.compute_control>`
method. You could also use the :meth:`step_callback <.BaseController.step_callback>` method to
update the controller state at runtime.

Note:
    You can only define one controller class in a single file. Otherwise we will not be able to
    determine which class to use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

import time

class BaseController(ABC):
    """Base class for controller implementations."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Instructions:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        self.initial_obs = initial_obs
        self.initial_info = initial_info
        self.close_ctrl_time = None
        self.return_pos = None
        self.return_status = 0
        self.finished = False


    @abstractmethod
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Instructions:
            Implement this method to return the target state to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] in absolute
            coordinates as a numpy array.
        """

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Callback function called once after the control step.

        You can use this function to update your controller's internal state, save training data,
        update your models, etc.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.
        """

    def episode_callback(self):
        """Callback function called once after each episode.

        You can use this function to reset your controller's internal state, save training data,
        train your models, compute additional statistics, etc.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.
        """

    def reset(self):
        """Reset internal variables if necessary."""

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""

    def close(self, obs):
        """Close the environment by stopping the drone and landing back at the starting position."""
        """Basically just a state machine that breaks, flys back and lands."""
        RETURN_HEIGHT = 1.75  # m
        BREAKING_DISTANCE = 1.0  # m
        BREAKING_DURATION = 3.0  # s
        RETURN_DURATION = 5.0  # s
        LAND_DURATION = 4.5  # s

        POS_CMD = 1
        KILL_CMD = -9

        try:  # prevent hanging process if drone not reachable
            if self.close_ctrl_time is None:
                self.return_status = 1
                print(f"close ctrl time None")
                self.close_ctrl_time = time.perf_counter()

                # Add a orhtogoonal part to the current drone velocity to the return pos so that the drones dont interfer with each other.
                self.return_pos = (
                    obs["pos"]
                    + (
                        obs["vel"] / (np.linalg.norm(obs["vel"]) + 1e-8) * 0.9
                        + obs["vel"]
                        * np.array([2 * self.controller_id - 1, -(2 * self.controller_id - 1), 0])
                        / (np.linalg.norm(obs["vel"]) + 1e-8)
                        * 0.1
                    )
                    * BREAKING_DISTANCE
                )
                self.return_pos[2] = RETURN_HEIGHT

                return (POS_CMD, np.array([*self.return_pos, 0, BREAKING_DURATION]))
            elif (
                time.perf_counter() - self.close_ctrl_time > BREAKING_DURATION - 1
                and self.return_status == 1
            ):
                print(f"close ctrl time past breaking duration")
                self.return_status = 2

                #self.return_pos[:2] = self.config.env.track.drones[self.controller_id].pos[:2]
                self.return_pos[:2] = self.initial_obs[:2]
                print(f"return position: {self.return_pos}")
                return (POS_CMD, np.array([*self.return_pos, 0, RETURN_DURATION]))
            elif (
                time.perf_counter() - self.close_ctrl_time > BREAKING_DURATION - 1 + RETURN_DURATION
                and self.return_status == 2
            ):
                self.return_status = 3
                self.return_pos[2] = 0
                # access to the landing function of the cf class is not implemented. Works this way too though.
                # self.cf.land(self.config.env.track.drone.pos[2], LAND_DURATION)
                return (POS_CMD, np.array([*self.return_pos, 0, LAND_DURATION]))

            elif (
                time.perf_counter() - self.close_ctrl_time
                > BREAKING_DURATION - 1 + RETURN_DURATION + LAND_DURATION
                and self.return_status == 3
            ):
                self.return_status = 4
                return (KILL_CMD, None)
            else:
                return None

        except Exception as e:
            print(f"error: {e}")
