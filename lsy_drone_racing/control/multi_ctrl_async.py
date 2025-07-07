from __future__ import annotations  # Python 3.10 type hints

import multiprocessing
#multiprocessing.set_start_method('spawn') # Or 'fork' or 'forkserver'
from munch import Munch

import time
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.constants import MASS
from inv_rl.attitude_mpc_wrapper import LearningController
from mpcc.control.controller_single import ControllerSingle as MPCC

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_controller_custom import AttitudeController as AttCtrl
import sys, os

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Define a sentinel for process shutdown
_SHUTDOWN_SIGNAL = "SHUTDOWN"

def _controller_worker_process_target(controller_class, obs_init, info_init, config_init, input_queue, output_queue):
    """
    Target function for each controller process.
    Instantiates the controller and processes commands from the input queue.
    """
    try:
        controller_instance = controller_class(obs_init, info_init, config_init)
        while True:
            # Wait for a command from the main process
            command, args, kwargs = input_queue.get()

            if command == _SHUTDOWN_SIGNAL:
                break # Exit the worker loop

            # Dispatch based on command
            if hasattr(controller_instance, command):
                method = getattr(controller_instance, command)
                result = method(*args, **kwargs)
                output_queue.put(result)
            else:
                raise AttributeError(f"Controller does not have method: {command}")
    except Exception as e:
        # Catch exceptions in the worker process and put them on the output queue
        # so the main process can be aware.
        print(f"Controller worker error ({controller_class.__name__}): {e}")
        output_queue.put(e) # Send the exception object back
        raise # Re-raise to let the process terminate and show traceback


class MultiController(Controller):
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
        # self.freq = config.env.freq
        # self.drone_mass = MASS
        # self.kp = np.array([0.4, 0.4, 1.25])
        # self.ki = np.array([0.05, 0.05, 0.05])
        # self.kd = np.array([0.2, 0.2, 0.4])
        # self.ki_range = np.array([2.0, 2.0, 0.4])
        # self.i_error = np.zeros(3)
        # self.g = 9.81
        self._tick = 0

        self._finished = False

        # --- Multiprocessing setup ---
        self.processes = []
        self.input_queues = []
        self.output_queues = []

        # Controller 0 Setup
        self.input_q_0 = multiprocessing.Queue()
        self.output_q_0 = multiprocessing.Queue()
        self.input_queues.append(self.input_q_0)
        self.output_queues.append(self.output_q_0)

        controller_class_0 = None
        current_info_0 = info.copy() # Make a copy to avoid modifying original info
        current_info_0["id"] = 0
        if "settings_controller0" in info.keys():
            controller_name = info["settings_controller0"]
            if controller_name == "pid":
                controller_class_0 = AttCtrl
            elif controller_name == "learning":
                controller_class_0 = LearningController
            else:
                raise NotImplementedError
        else:
            controller_class_0 = AttCtrl

        print(f"staring process 0")
        # Start process for controller_0
        process_0 = multiprocessing.Process(
            target=_controller_worker_process_target,
            args=(controller_class_0, obs, current_info_0, config, self.input_q_0, self.output_q_0)
        )
        process_0.start()
        self.processes.append(process_0)


        # Controller 1 Setup
        self.input_q_1 = multiprocessing.Queue()
        self.output_q_1 = multiprocessing.Queue()
        self.input_queues.append(self.input_q_1)
        self.output_queues.append(self.output_q_1)

        current_info_1 = info.copy() # Make a copy to avoid modifying original info
        current_info_1["id"] = 1
        controller_class_1 = MPCC # MPCC is always controller 1

        print(f"staring process 1")
        # Start process for controller_1
        process_1 = multiprocessing.Process(
            target=_controller_worker_process_target,
            args=(controller_class_1, obs, current_info_1, config, self.input_q_1, self.output_q_1)
        )
        process_1.start()
        self.processes.append(process_1)


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """CompuY9QU9Jte the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                 for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        # Send compute_control commands to both controllers in parallel
        self.input_q_0.put(("compute_control", (obs, info), {}))
        self.input_q_1.put(("compute_control", (obs, info), {}))


        # Wait for results from both controllers
        # We need to handle potential exceptions from the worker processes
        try:
            action_0, ctrl_info_0 = self.output_q_0.get()
            action_1, ctrl_info_1 = self.output_q_1.get()
        except Exception as e:
            print(f"Error retrieving results from controller processes: {e}")
            raise # Re-raise to propagate the error

        action = np.zeros((1, 2, 4))
        action[0, 0, :] = action_0
        action[0, 1, :] = action_1

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

        # Send step_callback commands to both controllers in parallel
        self.input_q_0.put(("step_callback", (action, obs, reward, terminated, truncated, info), {}))
        self.input_q_1.put(("step_callback", (action, obs, reward, terminated, truncated, info), {}))

        # Wait for results
        try:
            ctrl_finished_0 = self.output_q_0.get()
            ctrl_finished_1 = self.output_q_1.get()
        except Exception as e:
            print(f"Error retrieving step_callback results from controller processes: {e}")
            raise

        self._finished = ctrl_finished_1 # Assuming only controller_1 determines finished state
        return self._finished

    def episode_callback(self, **kwargs):
        """Reset the integral error."""
        self._tick = 0

        # Controller 0's episode_callback is called first, its output (X, U) is used by controller 1
        self.input_q_0.put(("episode_callback", (), kwargs))
        try:
            X, U = self.output_q_0.get()
        except Exception as e:
            print(f"Error retrieving episode_callback results from controller_0 process: {e}")
            raise

        # Now send to controller 1, using X and U from controller 0
        self.input_q_1.put(("episode_callback", (), {"X": X, "U": U}))
        try:
            # Assuming controller_1.episode_callback returns nothing important here
            _ = self.output_q_1.get()
        except Exception as e:
            print(f"Error retrieving episode_callback results from controller_1 process: {e}")
            raise
        return

    def episode_reset(self):
        """Reset the internal states of the controllers for a new episode."""
        # Send episode_reset commands to both controllers in parallel
        self.input_q_0.put(("episode_reset", (), {}))
        self.input_q_1.put(("episode_reset", (), {}))

        # Wait for results
        try:
            persistent_info_0 = self.output_q_0.get()
            persistent_info_1 = self.output_q_1.get()
        except Exception as e:
            print(f"Error retrieving episode_reset results from controller processes: {e}")
            raise

        return (persistent_info_0, persistent_info_1)

    def __del__(self):
        """Ensures all child processes are terminated when the object is garbage collected."""
        print("Shutting down controller processes...")
        for i, q_in in enumerate(self.input_queues):
            if self.processes[i].is_alive():
                q_in.put((_SHUTDOWN_SIGNAL, (), {})) # Send shutdown signal
                self.processes[i].join(timeout=1) # Give it a moment to shut down
                if self.processes[i].is_alive():
                    print(f"Process {i} did not terminate gracefully, attempting to terminate.")
                    self.processes[i].terminate() # Force terminate if still alive
        print("Controller processes shut down.")



if __name__ == "__main__":
        drone_mass = MASS
        kp = np.array([0.4, 0.4, 1.25])
        ki = np.array([0.05, 0.05, 0.05])
        kd = np.array([0.2, 0.2, 0.4])
        ki_range = np.array([2.0, 2.0, 0.4])
        i_error = np.zeros(3)
        g = 9.81
        _tick = 0

        _finished = False

        # --- Multiprocessing setup ---
        processes = []
        input_queues = []
        output_queues = []

        # Controller 0 Setup
        input_q_0 = multiprocessing.Queue()
        output_q_0 = multiprocessing.Queue()
        input_queues.append(input_q_0)
        output_queues.append(output_q_0)

        obs = {}
        info = {}
        config = Munch({"env": {"freq": 10}})

        controller_class_0 = None
        current_info_0 = info.copy() # Make a copy to avoid modifying original info
        current_info_0["id"] = 0
        if "settings_controller0" in info.keys():
            controller_name = info["settings_controller0"]
            if controller_name == "pid":
                controller_class_0 = AttCtrl
            elif controller_name == "learning":
                controller_class_0 = LearningController
            else:
                raise NotImplementedError
        else:
            controller_class_0 = AttCtrl

        print(f"staring process 0")
        # Start process for controller_0
        controller = MultiController(obs, info, config)
        process_0 = multiprocessing.Process(
            target=_controller_worker_process_target,
            args=(controller_class_0, obs, current_info_0, config, input_q_0, output_q_0)
        )
        time.sleep(10)
