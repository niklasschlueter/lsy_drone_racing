import os
from pathlib import Path
import fire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsy_drone_racing.utils.data_logging import DataVarIndex


class Plotter:
    """A class that plots the recorded data."""

    def __init__(self):
        self.file_path = None
        self.colors = ["b", "g", "r", "c", "m", "y"]

    def save_fig(self, fig, plot_name):
        plot_name = plot_name + ".png"
        file_path = self.save_path / plot_name

        # make sure directories exits
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        fig.savefig(file_path, dpi=200)
        print(f"figure saved to {file_path}")

    def plot_position(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure(figsize=(16, 9))
        ## X-Y Plane
        ax = fig.add_subplot(2, 2, 1)
        ax.set_title("XY")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_X.value],
            df.iloc[:, DataVarIndex.DES_POS_Y.value],
            "--",
            label="des",
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_X.value], df.iloc[:, DataVarIndex.POS_Y.value], label="meas"
        )
        ax.legend()

        ## X-Z Plane
        ax = fig.add_subplot(2, 2, 2)
        ax.set_title("XZ")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.axis("equal")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_X.value],
            df.iloc[:, DataVarIndex.DES_POS_Z.value],
            "--",
            label="des",
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_X.value], df.iloc[:, DataVarIndex.POS_Z.value], label="meas"
        )
        ax.legend()

        ## Y-Z Plane
        ax = fig.add_subplot(2, 2, 3)
        ax.set_title("YZ")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("Z [m]")
        ax.axis("equal")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_Y.value],
            df.iloc[:, DataVarIndex.DES_POS_Z.value],
            "--",
            label="des",
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_Y.value], df.iloc[:, DataVarIndex.POS_Z.value], label="meas"
        )
        ax.legend()

        ## 3d
        ax = fig.add_subplot(2, 2, 4, projection="3d")
        ax.set_title("3d Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_X.value],
            df.iloc[:, DataVarIndex.DES_POS_Y.value],
            df.iloc[:, DataVarIndex.DES_POS_Z.value],
            "--",
            label="des",
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_X.value],
            df.iloc[:, DataVarIndex.POS_Y.value],
            df.iloc[:, DataVarIndex.POS_Z.value],
            label="meas",
        )
        set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_velocity(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Velocity")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$v$ [m/s]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.VEL_X.value], "-", label=r"$v_{x, meas}$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_VEL_X.value], "--", label=r"$v_{x, des}$")

        ax.plot(timesteps, df.iloc[:, DataVarIndex.VEL_Y.value], "-", label=r"$v_{y, meas}$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_VEL_Y.value], "--", label=r"$v_{y, des}$")

        ax.plot(timesteps, df.iloc[:, DataVarIndex.VEL_Z.value], "-", label=r"$v_{z, meas}$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_VEL_Z.value], "--", label=r"$v_{z, des}$")
        ax.legend()

        return fig

    def plot_rpy(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure(figsize=(16, 9))

        ## rpy
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Angles (RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL.value], "-", label=r"roll")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH.value], "-", label=r"pitch")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW.value], "-", label=r"yaw")

        ax.legend()

        ## rpy rate
        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Angle Rates (RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle Rate [rad/s]")

        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL_RATE.value], "-", label=r"roll rate")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH_RATE.value], "-", label=r"pitch rate")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW_RATE.value], "-", label=r"yaw rate")

        ax.legend()

        return fig

    def plot_rpy_raw(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure(figsize=(16, 9))

        ## rpy
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Angle Vicon(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL_RAW.value], "-", label=r"roll")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH_RAW.value], "-", label=r"pitch")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW_RAW.value], "-", label=r"roll")

        ax.legend()

        ## rpy rate
        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Angle Estimator(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        # ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL.value], "-", label=r"roll")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH.value], "-", label=r"pitch")
        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.YAW.value], "-", label=r"roll"
        )  # its roll because diff. order!

        ax.legend()

        return fig

    def plot_position_raw(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure(figsize=(16, 9))

        ## rpy
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Position Estimator(m)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"[m]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_X.value], "-", label=r"roll")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_Y.value], "-", label=r"pitch")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_Z.value], "-", label=r"yaw")

        ax.legend()

        ## rpy rate
        ax = fig.add_subplot(3, 1, 2)
        ax.set_title("Position Vicon(m)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"[m]")

        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_RAW_X.value], "-", label=r"roll")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_RAW_Y.value], "-", label=r"pitch")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_RAW_Z.value], "-", label=r"yaw")

        ax.legend()

        ax = fig.add_subplot(3, 1, 3)
        ax.set_title("Error (m)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"[m]")

        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.POS_X.value] - df.iloc[:, DataVarIndex.POS_RAW_X.value],
            "-",
            label=r"roll",
        )
        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.POS_Y.value] - df.iloc[:, DataVarIndex.POS_RAW_Y.value],
            "-",
            label=r"pitch",
        )
        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.POS_Z.value] - df.iloc[:, DataVarIndex.POS_RAW_Z.value],
            "-",
            label=r"yaw",
        )

        ax.legend()

        return fig

    def plot_attitude_input(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        # check if attitude interface was actually used
        if df.iloc[1, DataVarIndex.CTRL_MODE] != "ATTITUDE":
            print(f"Skipping Attitude input plot, no attitude interface used.")
            return None

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Total Thrust Input")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$total thrust$ [N]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_THRUST], "-", label=r"$thrust_{des}$")
        ax.legend()

        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Angle Input")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$angle$ [rad]")

        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_ROLL.value], "-", label=r"$roll_{des}$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_PITCH.value], "-", label=r"$pitch_{des}$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.DES_YAW.value], "-", label=r"$yaw_{des}$")

        ax.legend()

        return fig

    def plot_data(
        self, file_path, show=False, save=False, save_path="plots/", create_dir_form_filename=True
    ):
        self.file_path = Path(file_path)
        self.show = show
        self.save = save
        self.save_path = Path(save_path)

        # create directory based on the data filename
        if create_dir_form_filename:
            self.save_path = self.save_path / self.file_path.stem

        figs = []
        plot_names = []

        figs.append(self.plot_position())
        plot_names.append("position")
        figs.append(self.plot_velocity())
        plot_names.append("velocity")
        figs.append(self.plot_rpy())
        plot_names.append("rpy")

        figs.append(self.plot_position_raw())
        plot_names.append("position_raw")
        figs.append(self.plot_rpy_raw())
        plot_names.append("rpy_raw")

        # only plot if attitude interface selected.
        fig = self.plot_attitude_input()
        if fig:
            figs.append(fig)
            plot_names.append("attitude_input")

        if self.save:
            [self.save_fig(fig, plot_name) for fig, plot_name in zip(figs, plot_names)]

        if self.show:
            plt.show()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__":
    # file_path = Path("data") / "last_run_sim.csv"
    # print(f"Using data file {file_path}")

    # plotter = Plotter()

    # plotter.plot_data(file_path, save=True, show=True)
    def plot_data_cli(prefix="sim", save=True, show=True):
        plotter = Plotter()
        file_path = Path("data") / f"last_run_{prefix}.csv"
        plotter.plot_data(file_path, save, show)

    fire.Fire(plot_data_cli)
