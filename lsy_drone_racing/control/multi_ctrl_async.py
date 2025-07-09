from __future__ import annotations

import multiprocessing
import pickle
from munch import Munch
import time
from typing import TYPE_CHECKING
import numpy as np

from crazyflow.constants import MASS
from inv_rl.attitude_mpc_wrapper import LearningController
from mpcc.control.controller_single import ControllerSingle as MPCC
from pympler import asizeof

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_controller_custom import AttitudeController as AttCtrl

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SHUTDOWN_SIGNAL = "SHUTDOWN"


def _controller_worker_process_target(controller_class, obs_init, info_init, config_init, conn):
    try:
        controller_instance = controller_class(obs_init, info_init, config_init)
        while True:
            while not conn.poll():
                pass
            command, args, kwargs = conn.recv()

            if command == _SHUTDOWN_SIGNAL:
                break

            if hasattr(controller_instance, command):
                method = getattr(controller_instance, command)
                result = method(*args, **kwargs)
                conn.send(result)
            else:
                raise AttributeError(f"Controller does not have method: {command}")
    except Exception as e:
        print(f"Worker error: {e}")
        conn.send(e)
        raise


class MultiController(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.processes = []
        self.parent_conns = []

        # --- Controller 0 ---
        parent_conn_0, child_conn_0 = multiprocessing.Pipe()
        self.parent_conns.append(parent_conn_0)

        controller_class_0 = AttCtrl if info.get("settings_controller0", "pid") == "pid" else LearningController

        process_0 = multiprocessing.Process(
            target=_controller_worker_process_target,
            args=(controller_class_0, obs, {**info, "id": 0}, config, child_conn_0)
        )
        process_0.start()
        self.processes.append(process_0)

        # --- Controller 1 ---
        parent_conn_1, child_conn_1 = multiprocessing.Pipe()
        self.parent_conns.append(parent_conn_1)

        process_1 = multiprocessing.Process(
            target=_controller_worker_process_target,
            args=(MPCC, obs, {**info, "id": 1}, config, child_conn_1)
        )
        process_1.start()
        self.processes.append(process_1)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        t0 = time.perf_counter()
        self.parent_conns[0].send(("compute_control", (obs, info), {}))
        self.parent_conns[1].send(("compute_control", (obs, info), {}))
        td = time.perf_counter()

        while not self.parent_conns[0].poll():
            pass
        action_0, ctrl_info_0 = self.parent_conns[0].recv()

        while not self.parent_conns[1].poll():
            pass
        action_1, ctrl_info_1 = self.parent_conns[1].recv()
        t1 = time.perf_counter()

        action = np.zeros((1, 2, 4))
        action[0, 0, :] = action_0
        action[0, 1, :] = action_1

        return action, (ctrl_info_0, ctrl_info_1)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self.parent_conns[0].send(("step_callback", (action, obs, reward, terminated, truncated, info), {}))
        self.parent_conns[1].send(("step_callback", (action, obs, reward, terminated, truncated, info), {}))

        while not self.parent_conns[0].poll():
            pass
        ctrl_finished_0 = self.parent_conns[0].recv()

        while not self.parent_conns[1].poll():
            pass
        ctrl_finished_1 = self.parent_conns[1].recv()

        self._finished = ctrl_finished_1
        return self._finished

    def episode_callback(self, **kwargs):
        self.parent_conns[0].send(("episode_callback", (), kwargs))
        X, U = self.parent_conns[0].recv()

        self.parent_conns[1].send(("episode_callback", (), {"X": X, "U": U}))
        _ = self.parent_conns[1].recv()
        return

    def episode_reset(self):
        self.parent_conns[0].send(("episode_reset", (), {}))
        self.parent_conns[1].send(("episode_reset", (), {}))

        while not self.parent_conns[0].poll():
            pass
        persistent_info_0 = self.parent_conns[0].recv()

        while not self.parent_conns[1].poll():
            pass
        persistent_info_1 = self.parent_conns[1].recv()
        return (persistent_info_0, persistent_info_1)

    def __del__(self):
        print("Shutting down controller processes...")
        for conn, proc in zip(self.parent_conns, self.processes):
            if proc.is_alive():
                conn.send((_SHUTDOWN_SIGNAL, (), {}))
                proc.join(timeout=1)
                if proc.is_alive():
                    proc.terminate()
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
        input_q_0 = Queue()
        output_q_0 = Queue()
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
