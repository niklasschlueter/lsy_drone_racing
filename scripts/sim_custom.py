"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

import numpy as np
import mujoco

from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils.utils import plot_mujoco_marker, render_trace, rotation_matrix_from_points

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 4,
    gui: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    # shape #runs, #horizon, #states
    X = []
    U = []
    traj_pos = None
    traj_rot = None

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        print(f"initial obs: {obs}")
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action, ctrl_info = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            # Synchronize the GUI.
            if config.sim.gui:
                if ((i * fps) % config.env.freq) < fps:
                    if i == 0:
                        env.unwrapped.sim.max_visual_geom = 10_000
                        traj_rot = []
                        traj_pos = ctrl_info["trajectory"].T
                        traj_rot = rotation_matrix_from_points(traj_pos[:-1, ...], traj_pos[1:, ...])
                        print(traj_rot.as_matrix().shape)
                        print(traj_pos.shape)
                        
                    if i > 1:
                        render_trace(env.unwrapped.sim.viewer, traj_pos, traj_rot)
                        #Uexit()
                    env.render()
            i += 1

            ## custom logging
            #r = R.from_quat(q)
            ## Convert to Euler angles in XYZ order
            #rpy = r.as_euler('xyz', degrees=False)  # Set degrees=False for radians
            #xcurrent = np.concatenate((obs["pos"], obs["vel"], rpy, np.array([last_f_collective, last_f_cmd]), last_rpy_cmd))
            ## overwrite last cmds
            #last_f_collective = f_collective
            #last_f_cmd = f_cmd
            #last_rpy_cmd = last_rpy_cmd


        x, u = controller.episode_callback()  # Update the controller internal state and models.
        X.append(np.array(x))
        U.append(np.array(u))
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Problem: Runs have different lenghts.
    # -> Quick fix: cut runs to length of run with shortest legths
    # TODO: Fix
    min_len_x = np.min([np.shape(x)[0] for x in X])
    min_len_u = np.min([np.shape(u)[0] for u in U])
    X = [x[:min_len_x, ...] for x in X]
    U = [u[:min_len_u, ...] for u in U]
    X = np.array(X)
    U = np.array(U)
    print(f"shape X: {np.shape(X)}")
    print(f"shape U: {np.shape(U)}")
    with open("data/X.pkl", "wb") as f:
        pickle.dump(X, f)
    with open("data/U.pkl", "wb") as f:
        pickle.dump(U, f)
    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
