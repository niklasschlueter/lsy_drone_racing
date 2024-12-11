"""The Vicon module provides an interface to the Vicon motion capture system for position tracking.

It defines the Vicon class, which handles communication with the Vicon system through ROS messages.
The Vicon class is responsible for:

* Tracking the drone and other objects (gates, obstacles) in the racing environment.
* Providing real-time pose (position and orientation) data for tracked objects.
* Calculating velocities and angular velocities based on pose changes.

This module is necessary to provide the real-world positioning data for the drone and race track
elements.
"""

from __future__ import annotations

import time
import logging

import numpy as np
import rospy
import yaml

# from crazyswarm.msg import StateVector
from vicon_bridge.msg import StateVector
from rosgraph import Master
from scipy.spatial.transform import Rotation as R
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty

from import_utils import get_ros_package_path

# logger = logging.getLogger("rosout." + __name__)


class ViconCustom:
    """Vicon interface for the pose estimation data for the drone and any other tracked objects.

    Vicon sends a stream of ROS messages containing the current pose data. We subscribe to these
    messages and save the pose data for each object in dictionaries. Users can then retrieve the
    latest pose data directly from these dictionaries.
    """

    def __init__(
        self,
        drone_names,
        track_names: list[str] = [],
        useKalman=False,
        # auto_track_drone: bool = True,
        timeout: float = 0.0,
    ):
        """Load the crazyflies.yaml file and register the subscribers for the Vicon pose data.

        Args:
            track_names: The names of any additional objects besides the drone to track.
            auto_track_drone: Infer the drone name and add it to the positions if True.
            timeout: If greater than 0, Vicon waits for position updates of all tracked objects
                before returning.
        """
        assert Master("/rosnode").is_online(), "ROS is not running. Please run hover.launch first!"
        try:
            rospy.init_node(f"drone_racing_vicon_listener")
        except rospy.exceptions.ROSException:
            ...  # ROS node is already running which is fine for us
        print(f"initalizing vicon listener for drone names: {drone_names}")
        self.drone_names = drone_names
        self.track_names = track_names
        self.useKalman = useKalman
        # Register the Vicon subscribers for the drone and any other tracked object
        self.pos: dict[str, np.ndarray] = {}
        self.rpy: dict[str, np.ndarray] = {}
        self.vel: dict[str, np.ndarray] = {}
        self.ang_vel: dict[str, np.ndarray] = {}
        self.time: dict[str, float] = {}

        self.pos_raw: dict[str, np.ndarray] = {}
        self.rpy_raw: dict[str, np.ndarray] = {}

        self.estimator_callback_counter = 0

        self.tf_sub = rospy.Subscriber("/tf", TFMessage, self.tf_callback)

        self.estimator_subs = []
        self.raw_vicon_data_subs = []
        for drone_name in drone_names:
            self.estimator_subs += [
                rospy.Subscriber(
                    f"/estimated_state_{drone_name}",
                    StateVector,
                    self.estimator_callback,
                    drone_name,
                )
            ]

            self.raw_vicon_data_subs += [
                rospy.Subscriber(
                    f"/vicon/{drone_name}/{drone_name}",
                    TransformStamped,
                    self.raw_vicon_callback,
                    drone_name,
                )
            ]

        # if useKalman:
        #    ### Kalman Estimator things.
        #    # TODO: Call service once: _flip_flag_for_outlier_test
        #    outlier_test_service_name = self.drone_name + "_flip_flag_for_outlier_test"

        #    # Wait for the service to be available
        #    rospy.wait_for_service(outlier_test_service_name)

        #    # Create a service proxy
        #    flip_flag = rospy.ServiceProxy(outlier_test_service_name, Empty)

        #    # Call the service at the start
        #    rospy.loginfo("Calling flip flag service at start.")
        #    flip_flag()

        if timeout:
            tstart = time.time()
            while not self.active and time.time() - tstart < timeout:
                time.sleep(0.01)
            if not self.active:
                raise TimeoutError(
                    "Timeout while fetching initial position updates for all tracked objects. "
                    f"Missing objects: {[k for k in self.track_names if k not in self.ang_vel]}"
                )
        time.sleep(0.1)

    def estimator_callback(self, data: StateVector, drone_name):
        """Save the drone state from the estimator node.

        Args:
            data: The StateVector message.
        """
        self.pos[drone_name] = np.array(data.pos)

        # Depending on whether we use the Kalman Filter or not,
        # we either get the rpy values from the quaternion or not.
        # TODO: Fix
        if self.useKalman:
            rpy = data.euler  # for kalman filter!
        else:
            rpy = R.from_quat(data.quat).as_euler("xyz")

        self.rpy[drone_name] = np.array(rpy)
        self.vel[drone_name] = np.array(data.vel)
        self.ang_vel[drone_name] = np.array(data.omega_b)

        # logger.info(f"Currently in process {mp.current_process()}, pos {self.pos}")
        self.estimator_callback_counter += 1

    def raw_vicon_callback(self, data: TFMessage, drone_name):
        """Save the position and orientation of all transforms.

        Args:
            data: The TF message containing the objects' pose.
        """
        # name = tf.child_frame_id.split("/")[-1]
        # Skip drone if it is also in track names, handled by the estimator_callback
        T, Rot = data.transform.translation, data.transform.rotation
        pos = np.array([T.x, T.y, T.z])
        rpy = R.from_quat([Rot.x, Rot.y, Rot.z, Rot.w]).as_euler("xyz")
        # self.time[name] = time.time()
        self.pos_raw[drone_name] = pos
        self.rpy_raw[drone_name] = rpy

    def tf_callback(self, data: TFMessage):
        """Save the position and orientation of all transforms.

        Args:
            data: The TF message containing the objects' pose.
        """
        for tf in data.transforms:
            name = tf.child_frame_id.split("/")[-1]
            # Skip drone if it is also in track names, handled by the estimator_callback
            if name in self.drone_names:
                continue
            if name not in self.track_names:
                continue
            T, Rot = tf.transform.translation, tf.transform.rotation
            pos = np.array([T.x, T.y, T.z])
            rpy = R.from_quat([Rot.x, Rot.y, Rot.z, Rot.w]).as_euler("xyz")
            self.time[name] = time.time()
            self.pos[name] = pos
            self.rpy[name] = rpy

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest pose of a tracked object.

        Args:
            name: The name of the object.

        Returns:
            The position and rotation of the object. The rotation is in roll-pitch-yaw format.
        """
        return self.pos[name], self.rpy[name]

    @property
    def poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest poses of all objects."""
        return np.stack(self.pos.values()), np.stack(self.rpy.values())

    @property
    def names(self) -> list[str]:
        """Get a list of actively tracked names."""
        return list(self.pos.keys())

    @property
    def active(self) -> bool:
        """Check if Vicon has sent data for each object."""
        # Check if drone is being tracked and if drone has already received updates
        for drone_name in self.drone_names:
            if drone_name not in self.pos:
                return False
        # Check remaining object's update status
        return all([name in self.pos for name in self.track_names])

    def close(self):
        """Unregister the ROS subscribers."""
        self.tf_sub.unregister()
        self.estimator_sub.unregister()
