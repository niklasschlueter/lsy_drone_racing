"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type
import pybullet as p

import numpy as np
import toml
from munch import munchify
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import BaseController

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from munch import Munch
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def map2pi(angle: NDArray[np.floating]) -> NDArray[np.floating]:
    """Map an angle or array of angles to the interval of [-pi, pi].

    Args:
        angle: Number or array of numbers.

    Returns:
        The remapped angles.
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def load_controller(path: Path) -> Type[BaseController]:
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
        subcls = inspect.isclass(mod) and issubclass(mod, BaseController)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, BaseController)]
    assert (
        len(controllers) > 0
    ), f"No controller found in {path}. Have you subclassed BaseController?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, BaseController)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> Munch:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The munchified config dict.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"
    with open(path, "r") as f:
        return munchify(toml.load(f))


def check_gate_pass(
    gate_idx: int,
    gate_pos: np.ndarray,
    gate_rot: R,
    gate_size: np.ndarray,
    drone_pos: np.ndarray,
    last_drone_pos: np.ndarray,
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the y-Axis (pointing from -y to +y). Therefore, we check if y
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the x
    and z box boundaries. First, we linearly interpolate to get the x and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        gate_pos: The position of the gate in the world frame.
        gate_rot: The rotation of the gate in the world frame.
        gate_size: The size of the gate box in meters.
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
    """
    # Transform last and current drone position into current gate frame.
    assert isinstance(gate_rot, R), "gate_rot has to be a Rotation object."
    last_pos_local = gate_rot.apply(last_drone_pos - gate_pos, inverse=True)
    pos_local = gate_rot.apply(drone_pos - gate_pos, inverse=True)
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    if last_pos_local[1] < 0 and pos_local[1] > 0:  # Drone has passed the goal plane
        alpha = -last_pos_local[1] / (pos_local[1] - last_pos_local[1])
        x_intersect = alpha * (pos_local[0]) + (1 - alpha) * last_pos_local[0]
        z_intersect = alpha * (pos_local[2]) + (1 - alpha) * last_pos_local[2]
        # Divide gate size by 2 to get the distance from the center to the edges
        if abs(x_intersect) < gate_size[0] / 2 and abs(z_intersect) < gate_size[1] / 2:
            print(
                f"Drone successfully passed gate {gate_idx} with distance {x_intersect, z_intersect}"
            )
            return True
        print(f"Drone missed gate {gate_idx} with distance {x_intersect, z_intersect}")
        print(f"believed gate pos: {gate_pos}")
        print(f"believed drone pos: {drone_pos}")
    return False


def draw_trajectory(
    initial_info: dict,
    waypoints: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_z: np.ndarray,
    num_plot_points: int = 50,
    color=(1, 0, 0, 1),
):
    """Draw a trajectory in PyBullet's GUI."""
    for point in waypoints:
        sphere_pos = point
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=sphere_pos)

    step = max(int(ref_x.shape[0] / num_plot_points), 1)
    for i in range(step, ref_x.shape[0], step):
        p.addUserDebugLine(
            lineFromXYZ=[ref_x[i - step], ref_y[i - step], ref_z[i - step]],
            lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
            lineWidth=5,
            lineColorRGB=color[:3],
            # physicsClientId=initial_info["pyb_client"],
        )
    p.addUserDebugLine(
        lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
        lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
        lineWidth=5,
        lineColorRGB=color[:3],
        # physicsClientId=initial_info["pyb_client"],
    )


def draw_segment_of_traj(
    initial_info: dict, start_point, end_point, color=(1, 0, 0, 1), lineWidth=5, lifeTime=0
):
    """Draw line between two points using PyBullet."""

    p.addUserDebugLine(
        lineFromXYZ=start_point,
        lineToXYZ=end_point,
        lineWidth=lineWidth,
        lineColorRGB=color[:3],
        lifeTime=lifeTime,
        # physicsClientId=initial_info["pyb_client"],
    )
    return
