import json
import pandas as pd
from enum import IntEnum, unique
#import wandb
import numpy as np
import csv
import time
import os
from scipy.spatial.transform import Rotation as R

DataVarIndex = IntEnum(
    "DataVarIndex",
    [
        "TIME",
        "POS_X",
        "POS_Y",
        "POS_Z",
        "VEL_X",
        "VEL_Y",
        "VEL_Z",
        "q1", 
        "q2", 
        "q3", 
        "q4", 
        "ROLL",
        "PITCH",
        "YAW",
        "ROLL_RATE",
        "PITCH_RATE",
        "YAW_RATE",
        "DES_POS_X",
        "DES_POS_Y",
        "DES_POS_Z",
        "DES_YAW",
        "DES_VEL_X",
        "DES_VEL_Y",
        "DES_VEL_Z",
        "DES_THRUST",
        "DES_ROLL",
        "DES_PITCH",
        "CTRL_MODE",
        "POS_RAW_X",
        "POS_RAW_Y",
        "POS_RAW_Z",
        "ROLL_RAW",
        "PITCH_RAW",
        "YAW_RAW",
    ],
    start=0,
)


class DataLogger:
    """A class that logs the recorded data to a csv file using the DataVarIndex."""

    def __init__(self, filename, ctrl_mode):
        self.filename = filename
        self.create_csv()
        self.start_time = None
        if ctrl_mode not in ["attitude", "state"]:
            raise ValueError
        self.ctrl_mode = ctrl_mode

    def create_csv(self, drone_index=0):
        filename = None
        if drone_index is not None:
            filename = (
                self.filename.split(".")[0] + "_" + str(drone_index) + "." + self.filename.split(".")[1]
            )
        else:
            filename = self.filename
        # Create folders if necessary
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Create the csv file
        with open(filename, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow([var.name for var in DataVarIndex])

    def _write_data(self, data, drone_index=0):
        """Log the data to the csv file."""
        # Make sure that the data has the correct length
        assert len(data) == len(DataVarIndex)

        # Log the data
        filename = None
        if drone_index is not None:
            filename = (
                self.filename.split(".")[0] + "_" + str(drone_index) + "." + self.filename.split(".")[1]
            )
        else:
            filename = self.filename
        with open(filename, mode="a") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def log_data(self, obs, action, drone_index=0):
        # TODO: Make logging dependent on action type.
        # set start time
        if self.start_time is None:
            self.start_time = time.perf_counter()

        pos = obs["pos"]
        vel = obs["vel"]
        quat = obs["quat"]

        r = R.from_quat(quat)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler('xyz', degrees=False)  # Set degrees=False for radians
        #rpy = obs["rpy"]
        ang_vel = obs["ang_vel"]

        data = [None] * len(DataVarIndex)

        data[DataVarIndex.TIME] = time.perf_counter() - self.start_time
        data[DataVarIndex.POS_X] = pos[0]
        data[DataVarIndex.POS_Y] = pos[1]
        data[DataVarIndex.POS_Z] = pos[2]

        data[DataVarIndex.VEL_X] = vel[0]
        data[DataVarIndex.VEL_Y] = vel[1]
        data[DataVarIndex.VEL_Z] = vel[2]

        data[DataVarIndex.q1] = quat[0]
        data[DataVarIndex.q2] = quat[1]
        data[DataVarIndex.q3] = quat[2]
        data[DataVarIndex.q4] = quat[3]

        data[DataVarIndex.ROLL] = rpy[0]
        data[DataVarIndex.PITCH] = rpy[1]
        data[DataVarIndex.YAW] = rpy[2]

        data[DataVarIndex.ROLL_RATE] = ang_vel[0]
        data[DataVarIndex.PITCH_RATE] = ang_vel[1]
        data[DataVarIndex.YAW_RATE] = ang_vel[2]

        # only log raw vicon data if available
        if "pos_raw" in obs.keys():
            pos_raw = obs["pos_raw"]
            data[DataVarIndex.POS_RAW_X] = pos_raw[0]
            data[DataVarIndex.POS_RAW_Y] = pos_raw[1]
            data[DataVarIndex.POS_RAW_Z] = pos_raw[2]

            rpy_raw = obs["rpy_raw"]
            data[DataVarIndex.ROLL_RAW] = rpy_raw[0]
            data[DataVarIndex.PITCH_RAW] = rpy_raw[1]
            data[DataVarIndex.YAW_RAW] = rpy_raw[2]

        # position interface inputs
        if self.ctrl_mode == "state":
            data[DataVarIndex.CTRL_MODE] = "FULL"
            target_pos, target_vel, target_acc, target_yaw, target_rpy_rate = (
                action[:3],
                action[3:6],
                action[6:9],
                action[9],
                action[10:],
            )

            data[DataVarIndex.DES_POS_X] = target_pos[0]
            data[DataVarIndex.DES_POS_Y] = target_pos[1]
            data[DataVarIndex.DES_POS_Z] = target_pos[2]

            data[DataVarIndex.DES_YAW] = float(target_yaw)  # make target_yaw a float

            data[DataVarIndex.DES_VEL_X] = target_vel[0]
            data[DataVarIndex.DES_VEL_Y] = target_vel[1]
            data[DataVarIndex.DES_VEL_Z] = target_vel[2]

        elif self.ctrl_mode == "attitude":
            data[DataVarIndex.CTRL_MODE] = "ATTITUDE"
            thrust_des, rpy_des = action[0], action[1:]

            data[DataVarIndex.DES_THRUST] = thrust_des
            data[DataVarIndex.DES_ROLL] = rpy_des[0]
            data[DataVarIndex.DES_PITCH] = rpy_des[1]
            data[DataVarIndex.DES_YAW] = rpy_des[2]

        self._write_data(data, drone_index=drone_index)


# def load_data(filename):
#    """Load the data from the csv file and return it as a numpy array."""
#    # Read the data from the csv file, skipping the first row
#    # and the last column has to be transformed using Status enum
#    pd_data = pd.read_csv(filename)
#    pd_data[DataVarIndex.STATUS.name] = pd_data[DataVarIndex.STATUS.name].apply(
#        lambda s: match_status[s]
#    )
#    data = pd_data.to_numpy()#
#
#    # There may be a mismatch in the number of columns and the number of DataVarIndex. Add dummy values for the missing columns
#    num_columns = len(DataVarIndex)
#    num_data_columns = data.shape[1]
#    dummy_data = np.zeros((data.shape[0], num_columns - num_data_columns))
#    data = np.hstack((data, dummy_data))
#
#    return data