"""Simulate a multi-drone race.

Run as:

    $ python scripts/multi_sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from mpcc.logging.data_logging import DataLogger

from lsy_drone_racing.control.attitude_controller_custom import AttitudeController
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils.utils import render_trace, _rotation_matrix_from_points

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.multi_drone_race import MultiDroneRacingEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "exp_prediction_error.toml",
    controller: str | None = None,
    n_runs: int = 5,
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
    logger.warning(
        "The simulation currently only supports running with one controller type and one set of "
        "environment parameters (i.e. frequencies, control mode etc.). Only using the settings for "
        "the first drone."
    )
    # Load the controller module
    if controller is None:
        controller = config.controller[0]["file"]
    controller_path = Path(__file__).parents[1] / "lsy_drone_racing/control" / controller
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: MultiDroneRacingEnv = gymnasium.make(
        "MultiDroneRacing-v0",
        freq=config.env.kwargs[0]["freq"],
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.kwargs[0]["sensor_range"],
        control_mode=config.env.kwargs[0]["control_mode"],
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    # We use the same example controllers for this script as for the single-drone case. These expect
    # the config to have env.freq set, so we copy it here. Actual multi-drone controllers should not
    # rely on this.
    config.env.freq = config.env.kwargs[0]["freq"]
    env = JaxToNumpy(env)
    n_drones, n_worlds = env.unwrapped.sim.n_drones, env.unwrapped.sim.n_worlds

    # If we want to retain information between episodes.
    # controller = None
    #tau = 1
    repetitions = 10
    no_runs = 2
    head_start_times = np.random.uniform(1.5, 1.5, repetitions*n_runs)
    print(f"head start times: {head_start_times}")
    for rep in range(repetitions):
        for opponent_ctrl in ["learning"]:#, "pid"]:#[::-1]:#, "pid"]:
            for predictor, n_runs in zip(["linear", "learning", "acados"], [0, no_runs, 0]):
                persistent_info = ({}, {})
                # Consecutive Repetitions (only makes sense for learning between episodes)
                for n_run in range(n_runs):  # Run n_runs episodes with the controller
                    print(f"STARTING RUN {n_run}/{n_runs} WITH OPP CTRL {opponent_ctrl} AND PREDICTOR {predictor} IN REPETITION {rep}")
                    obs, info = env.reset()
                    #PID_T_MIN, PID_T_MAX = 1.4, 2.0
                    #MPCC_T_MIN, MPCC_T_MAX = 0.85, 0.95
                    # For information that is persistent between episodes.
                    #info["persistent"] = persistent_info
                    #info["PID_time_scaling"] = (PID_T_MAX - PID_T_MIN) * tau + PID_T_MIN
                    #info["MPCC_weight_scale"] = (MPCC_T_MAX - MPCC_T_MIN) * tau + MPCC_T_MIN
                    # Choose controller!
                    #opponent_ctrl = "learning"#"pid"
                    info["settings_controller0"] = opponent_ctrl
                    #info["settings_controller1"] = "mpcc"

                    #predictor = "linear" #"learning" # "linear", "acados"
                    info["settings_predictor"] = predictor 
                    #info["settings_controller1"] = "mpcc"

                    hover_time = head_start_times[(rep+1)*(n_run+1)-1]
                    print("#################################################################")
                    print("#################################################################")
                    print("#################################################################")
                    print(f"hover time exp pred: {hover_time}")
                    print("#################################################################")
                    print("#################################################################")
                    print("#################################################################")
                    info["settings_initial_hover_time"] = hover_time

                    info["persistent"] = persistent_info
                    #info["PID_time_scaling"] = 2.0
                    #info["MPCC_weight_scale"] = 1.0
                    # Pass the episode number.
                    info["n_run"] = n_run
                    controller: Controller = controller_cls(obs, info, config)
                    #opponent_ctrl = (
                    #    "pid" if isinstance(controller.controller_0, AttitudeController) else "mpcc"
                    #)

                    #prediction = controller.controller_1.params.MPC_solver.opponent_prediction
                    save_path = Path(__file__).parents[1] / "saves/exp_prediction_error" / opponent_ctrl
                    # save_path = Path(__file__).parents[1] / "saves/debug" / opponent_ctrl
                    save_path = save_path / predictor / f"{rep:.1f}"
                    save_path.mkdir(exist_ok=True, parents=True)
                    i = 0
                    fps = 30
                    controller.controller_1.data_logger = DataLogger(
                        str(save_path / f"run{n_run:03d}.csv"), "attitude"
                    )

                    while True:
                        curr_time = i / config.env.freq
                        action, ctrl_info = controller.compute_control(obs, info)
                        obs, reward, terminated, truncated, info = env.step(action)
                        # Update the controller internal state and models.
                        controller_finished = controller.step_callback(
                            action, obs, reward, terminated, truncated, info
                        )
                        done = terminated | truncated | controller_finished
                        # Synchronize the GUI.
                        if config.sim.gui:
                            if ((i * fps) % config.env.freq) < fps:
                                if i == 0:
                                    # Set number of visual elements in sim
                                    env.unwrapped.sim.max_visual_geom = 1_000
                                    # Create arrays for the trajectories we want to plot
                                    traj_pos = []
                                    traj_rot = []
                                    # target len of the trajectory we want to plot (number of points)
                                    target_len = 100
                                    # Different traj. colors for different drones
                                    colors = ([1, 0, 0, 0.5], [0, 0, 1, 0.5])
                                    for _id, color in zip(range(n_drones), colors):
                                        # resample traj. such that it has the length traget_len
                                        traj = ctrl_info[_id]["trajectory"]
                                        indices = np.linspace(0, len(traj) - 1, target_len).astype(int)
                                        traj = traj[indices]
                                        # Calculate all the things to be able to plot the trajectory
                                        traj_pos.append(traj)  # ctrl_info[_id]["trajectory"])#.T
                                        traj_rot.append(
                                            _rotation_matrix_from_points(
                                                traj_pos[-1][:-1, ...], traj_pos[-1][1:, ...]
                                            )
                                        )
                                if i > 1:
                                    # Render the trajectory
                                    for _id, color in zip(range(n_drones), colors):
                                        # Calculate all the things to be able tot plot the trajectory
                                        render_trace(
                                            env.unwrapped.sim.viewer, traj_pos[_id], traj_rot[_id], color
                                        )

                                        # Render the horizon
                                        if len(ctrl_info[_id]["horizon"]) > 1:
                                            horiz_pos = ctrl_info[_id]["horizon"][:, :3]
                                            horiz_rot = _rotation_matrix_from_points(
                                                horiz_pos[:-1, ...], horiz_pos[1:, ...]
                                            )
                                            render_trace(
                                                env.unwrapped.sim.viewer,
                                                horiz_pos,
                                                horiz_rot,
                                                color=[0.0, 1.0, 0.0, 1.0],
                                            )

                                        # Render opp prediction
                                        if len(ctrl_info[_id]["opp_prediction"]) > 1:
                                            horiz_pos = ctrl_info[_id]["opp_prediction"][:, :3]
                                            horiz_rot = _rotation_matrix_from_points(
                                                horiz_pos[:-1, ...], horiz_pos[1:, ...]
                                            )
                                            render_trace(
                                                env.unwrapped.sim.viewer,
                                                horiz_pos,
                                                horiz_rot,
                                                color=[1.0, 1.0, 0.0, 1.0],
                                            )

                                env.render()
                        i += 1
                        if done:
                            break

                    controller.episode_callback()  # Update the controller internal state and models.
                    log_episode_stats(obs, info, config, curr_time)
                    persistent_info = controller.episode_reset()

    # Close the environment
    env.close()


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    finished = gates_passed == -1
    logger.info((f"Flight time (s): {curr_time}\nDrones finished: {finished}\n"))


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
