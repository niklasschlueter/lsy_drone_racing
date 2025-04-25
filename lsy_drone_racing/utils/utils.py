"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type
import numpy as np
import mujoco
import einops

import toml
from ml_collections import ConfigDict

from lsy_drone_racing.control.controller import Controller
from scipy.spatial.transform import Rotation as R
 

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


def plot_mujoco_marker(env, pos, size=np.array([0.03, 0.03, 0.03]), rgba=np.array([0.8, 0.2, 0.2, 1.0])):
    env.unwrapped.sim.viewer.viewer.add_marker(
    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size, pos=pos, rgba=rgba)


def render_trace(viewer, pos, rot, color=[1.0, 0.0, 0.0, 1.0]):
    """Render traces of the drone trajectories."""
    if len(pos) < 2 or viewer is None:
        return
    
    assert isinstance(pos, np.ndarray)
    n_trace = len(rot)
    sizes = np.zeros((n_trace, 3))
    sizes[..., 2] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    sizes[..., :2] = 20.0
    mats = rot.as_matrix()

    for i in range(n_trace):
        viewer.viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=sizes[i],
            pos=pos[i],
            mat=mats[i].flatten(),
            rgba=np.array(color),
        )

def rotation_matrix_from_points(p1, p2 ) -> R:
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))
