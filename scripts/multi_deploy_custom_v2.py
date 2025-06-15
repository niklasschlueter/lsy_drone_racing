"""Launch script for the real race with multiple drones.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
import numpy as np

import fire
import gymnasium
import rclpy

from lsy_drone_racing.utils import load_config, load_controller
import jax
jax.config.update("jax_platform_name", "cpu")

if TYPE_CHECKING:
    from multiprocessing.synchronize import Barrier

    from ml_collections import ConfigDict

    from lsy_drone_racing.envs.real_race_env import RealMultiDroneRaceEnv

logger = logging.getLogger(__name__)


def control_loop(rank: int, config: ConfigDict, start_barrier: Barrier):
    """Control loop for the drone."""
    rclpy.init()  # Start the ROS library
    node = rclpy.create_node(f"drone{rank}")
    # Override the env config with the kwargs for this particular drone
    config.env.freq = config.env.kwargs[rank]["freq"]
    config.env.sensor_range = config.env.kwargs[rank]["sensor_range"]
    config.env.control_mode = config.env.kwargs[rank]["control_mode"]
    logger.error(f"Starting drone {rank}")
    env: RealMultiDroneRaceEnv = gymnasium.make(
        "RealMultiDroneRacing-v0",
        drones=config.deploy.drones,
        rank=rank,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
    )
    logger.error(
        f"Drone {rank} env created with controller {config.controller[rank]['file']}"
        f" for drone {config.deploy.drones[rank]['id']} on channel {config.deploy.drones[rank]['channel']}"
    )
    try:
        options = {
            "check_drone_start_pos": config.deploy.check_drone_start_pos,
            "check_race_track": config.deploy.check_race_track,
            "real_track_objects": config.deploy.real_track_objects,
        }
        obs, info = env.reset(options=options)
        info["id"] = rank
        info["n_run"] = 0 # TODO: Use this 


        persistent_info = ({}, {}) # TODO: Use this
        try:
            with open('persistent.pkl', 'rb') as f:
                print(f"loading persistent dict!")
                persistent_dic_0= pickle.load(f)
            persistent_info = (persistent_dic_0, {}) # TODO: Use this
        except:
            print(f"could load persistent info!")


        info["persistent"] = persistent_info
        # TODO: Specify log path more clearly
        info["log_path"] = Path(__file__).parents[1] / "saves/deploy"/ f"run{info['n_run']:03d}.csv"
        next_obs = obs  # Set next_obs to avoid errors when the loop never enters

        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        # Will take the absolute path if provided in config.controller.file
        controller_path = control_path / config.controller[rank]["file"]
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)

        start_barrier.wait(timeout=30.0)  # Wait for all drones to be ready at the same time
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            action, _ = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(action, next_obs, reward, terminated, truncated, info)
            #print(f"Race time: {time.perf_counter() - start_time:.3f}s")
            if terminated or truncated or controller_finished:
                if controller_finished:
                    print(f"contoller finished! - terminating because of this")
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                node.get_logger().warning(
                    f"Controller {rank} exceeded loop frequency by {exc:.3f}s",
                    throttle_duration_sec=2,
                )
        ep_time = time.perf_counter() - start_time
        success = (next_obs["target_gate"] == -1)[rank]
        drone_id = config.deploy.drones[rank]["id"]
        drone_controller = config.controller[rank]["file"]
        print(
            f"Drone {drone_id} with controller {drone_controller} finished track: {success} \n"
            + (f"Track time: {ep_time:.3f}s" if success else "Task not completed")
        )
        print(f"Calling episode callback!")
        # The learning controller should throw an assert error here, because he has no data.
        # The non-learning controller should save data here into U_sim.pkl and X_sim.pkl
        controller.episode_callback()
    finally:
        node.destroy_node()
        env.close()


def main(config: str = "deploy_v2.toml"):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    n_drones = len(config.deploy.drones)
    assert len(config.controller) == n_drones, "Number of drones and controllers must match."
    assert len(config.env.kwargs) == n_drones, "Number of drones and env kwargs must match."
    assert len(config.env.track.drones) == n_drones, "Number of drones and track drones must match."
    n_drones = len(config.controller)
    ctx = mp.get_context("spawn")
    start_barrier = ctx.Barrier(n_drones)
    drone_processes = [
        ctx.Process(target=control_loop, args=(i, config, start_barrier)) for i in range(n_drones)
    ]
    for p in drone_processes:
        p.start()

    while any(p.is_alive() for p in drone_processes):
        time.sleep(0.2)

    if True:

        print(f"After the controller run!")
        with open('U_sim.pkl', 'rb') as f:
            U = pickle.load(f)

        with open('X_sim.pkl', 'rb') as f:
            X = pickle.load(f)

        print(f"shape X: {np.shape(X)}")
        print(f"shape U: {np.shape(U)}")

        with open('training_info.pkl', 'rb') as f:
            training_dic = pickle.load(f)
        print(f"training dic: {training_dic}")

        from inv_rl.trainer import train
        # Create a new leanring controller 
        #control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        ## Will take the absolute path if provided in config.controller.file
        #controller_path = control_path / config.controller[1]["file"]
        #controller_cls = load_controller(controller_path)
        #controller = controller_cls(obs, info, config)

        try: 
            with open('persistent.pkl', 'rb') as f:
                persistent = pickle.load(f)

            X_train = persistent["X"]
            U_train = persistent["U"]
            print(f"loading data from persistent!")
        except:
            X_train = []
            U_train = []
            print(f"No persistent found - no data loaded!")

        X_train += [X]
        U_train += [U]
        print(f"num trajectories in data: {len(X_train)}")
        td = training_dic
        w = td["w"] 
        print(f"w prev: {w}")
        batch_size = td["batch_size"]
        epochs = td["epochs"]
        lr = td["lr"]
        dt = td["dt"]
        positions = td["positions"]
        traj_length = td["traj_length"]
        num_batches = td["num_batches"]
        w = train(X_train, U_train, w, batch_size, epochs, lr, dt, positions, traj_length, M=100, num_batches=num_batches)
        print(f"w new: {w}")

        persistent_dict = {"w": w, "X": X_train, "U": U_train}
        with open("persistent.pkl", "wb") as f:
            pickle.dump(persistent_dict, f)








if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)