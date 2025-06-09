import os
from pathlib import Path
import fire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import scienceplots

import matplotlib.ticker as ticker

plt.style.use(['science','ieee'])        

plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "Times New Roman",
    #"font.size": 12,
    #"pgf.texsystem": "pdflatex",
    "axes.grid": True,
    "figure.figsize": (7.16, 4.5),
    "figure.dpi": 400,
})

#plt.rcParams['figure.figsize'] = (7.16, 4.5)  # for double-column plot
#plt.rcParams['figure.dpi'] = 300
#plt.style.use('science')        
#def_fig_size = (7.16,4.5)
#plt.rcParams['axes.grid'] = True


from mpcc.logging.data_logging import DataVarIndex

def load_csvs_as_dfs_from_folder(path: Path) -> list[pd.DataFrame]:
    # Path to the directory containing prediction error data
    # Find the most recent file(s) in the directory
    files = list(path.glob("*.csv"))
    print(f"files found: {files}")
    if not files:
        raise RuntimeError(f"No CSV files found in {path}")
    # Sort files by modification time (most recent first)
    files.sort(key=lambda x: x.stat().st_mtime)
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    return dfs

class Plotter:
    """A class that plots the recorded data."""

    def __init__(self):
        self.file_path = None
        self.colors = ["b", "g", "r", "c", "m", "y"]

    def save_fig(self, fig, plot_name):
        plot_name = plot_name + ".pdf"
        file_path = self.save_path / plot_name

        # make sure directories exits
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        fig.savefig(file_path, dpi=200)
        print(f"figure saved to {file_path}")

    def plot_position(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        
        #fig = plt.figure(figsize=(16, 9))
        fig = plt.figure()
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
            df.iloc[:, DataVarIndex.POS_X.value],
            df.iloc[:, DataVarIndex.POS_Y.value],
            label="meas",
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
            df.iloc[:, DataVarIndex.POS_X.value],
            df.iloc[:, DataVarIndex.POS_Z.value],
            label="meas",
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
            df.iloc[:, DataVarIndex.POS_Y.value],
            df.iloc[:, DataVarIndex.POS_Z.value],
            label="meas",
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

    #def plot_position_with_gates(self):
    #    # Read the data from the csv file
    #    df = pd.read_csv(self.file_path)

    #    gate_1 = np.array([df.iloc[0, DataVarIndex.GATE_POS_X_1.value], df.iloc[0, DataVarIndex.GATE_POS_Y_1.value], df.iloc[0, DataVarIndex.GATE_POS_Z_1.value]])
    #    gate_2 = np.array([df.iloc[0, DataVarIndex.GATE_POS_X_2.value], df.iloc[0, DataVarIndex.GATE_POS_Y_2.value], df.iloc[0, DataVarIndex.GATE_POS_Z_2.value]])
    #    gate_3 = np.array([df.iloc[0, DataVarIndex.GATE_POS_X_3.value], df.iloc[0, DataVarIndex.GATE_POS_Y_3.value], df.iloc[0, DataVarIndex.GATE_POS_Z_3.value]])
    #    gate_4 = np.array([df.iloc[0, DataVarIndex.GATE_POS_X_4.value], df.iloc[0, DataVarIndex.GATE_POS_Y_4.value], df.iloc[0, DataVarIndex.GATE_POS_Z_4.value]])

    #    #rpy_1 = np.array([df.iloc[0, DataVarIndex.GATE_R_1.value], df.iloc[0, DataVarIndex.GATE_P_1.value], df.iloc[0, DataVarIndex.GATE_Y_1.value]])
    #    #rpy_2 = np.array([df.iloc[0, DataVarIndex.GATE_R_2.value], df.iloc[0, DataVarIndex.GATE_P_2.value], df.iloc[0, DataVarIndex.GATE_Y_2.value]])
    #    #rpy_3 = np.array([df.iloc[0, DataVarIndex.GATE_R_3.value], df.iloc[0, DataVarIndex.GATE_P_3.value], df.iloc[0, DataVarIndex.GATE_Y_3.value]])
    #    #rpy_4 = np.array([df.iloc[0, DataVarIndex.GATE_R_4.value], df.iloc[0, DataVarIndex.GATE_P_4.value], df.iloc[0, DataVarIndex.GATE_Y_4.value]])


    #    # Extract gate yaw angles (only yaw is relevant)
    #    yaw_1 = df.iloc[0, DataVarIndex.GATE_Y_1.value]
    #    yaw_2 = df.iloc[0, DataVarIndex.GATE_Y_2.value]
    #    yaw_3 = df.iloc[0, DataVarIndex.GATE_Y_3.value]
    #    yaw_4 = df.iloc[0, DataVarIndex.GATE_Y_4.value]
    #    


    #    # Function to plot a gate line
    #    def plot_gate(ax, pos, yaw, length=0.2):
    #        dx = np.cos(yaw) * length
    #        dy = np.sin(yaw) * length
    #        start = [pos[0] - dx, pos[1] - dy]
    #        end = [pos[0] + dx, pos[1] + dy]
    #        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)



    #    plt.plot


    #    fig = plt.figure(figsize=(16, 9))
    #    ## X-Y Plane
    #    ax = fig.add_subplot(2, 2, 1)
    #    ax.set_title("XY")
    #    ax.set_xlabel("X [m]")
    #    ax.set_ylabel("Y [m]")
    #    ax.axis("equal")
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.DES_POS_X.value],
    #        df.iloc[:, DataVarIndex.DES_POS_Y.value],
    #        "--",
    #        label="des",
    #    )
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.POS_X.value],
    #        df.iloc[:, DataVarIndex.POS_Y.value],
    #        label="meas",
    #    )

    #    # Plot gates as black lines
    #    plot_gate(ax, gate_1, yaw_1)
    #    plot_gate(ax, gate_2, yaw_2)
    #    plot_gate(ax, gate_3, yaw_3)
    #    plot_gate(ax, gate_4, yaw_4)
    #    ax.legend()

    #    ## X-Z Plane
    #    ax = fig.add_subplot(2, 2, 2)
    #    ax.set_title("XZ")
    #    ax.set_xlabel("X [m]")
    #    ax.set_ylabel("Z [m]")
    #    ax.axis("equal")
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.DES_POS_X.value],
    #        df.iloc[:, DataVarIndex.DES_POS_Z.value],
    #        "--",
    #        label="des",
    #    )
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.POS_X.value],
    #        df.iloc[:, DataVarIndex.POS_Z.value],
    #        label="meas",
    #    )
    #    ax.legend()

    #    ## Y-Z Plane
    #    ax = fig.add_subplot(2, 2, 3)
    #    ax.set_title("YZ")
    #    ax.set_xlabel("Y [m]")
    #    ax.set_ylabel("Z [m]")
    #    ax.axis("equal")
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.DES_POS_Y.value],
    #        df.iloc[:, DataVarIndex.DES_POS_Z.value],
    #        "--",
    #        label="des",
    #    )
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.POS_Y.value],
    #        df.iloc[:, DataVarIndex.POS_Z.value],
    #        label="meas",
    #    )
    #    ax.legend()

    #    ## 3d
    #    ax = fig.add_subplot(2, 2, 4, projection="3d")
    #    ax.set_title("3d Trajectory")
    #    ax.set_xlabel("X [m]")
    #    ax.set_ylabel("Y [m]")
    #    ax.set_zlabel("Z [m]")
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.DES_POS_X.value],
    #        df.iloc[:, DataVarIndex.DES_POS_Y.value],
    #        df.iloc[:, DataVarIndex.DES_POS_Z.value],
    #        "--",
    #        label="des",
    #    )
    #    ax.plot(
    #        df.iloc[:, DataVarIndex.POS_X.value],
    #        df.iloc[:, DataVarIndex.POS_Y.value],
    #        df.iloc[:, DataVarIndex.POS_Z.value],
    #        label="meas",
    #    )
    #    set_axes_equal(ax)

    #    ax.legend()

    #    return fig
    def plot_position_with_gates(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        df = pd.read_csv(self.file_path)

        gate_positions = [
            np.array([df.iloc[0, DataVarIndex.GATE_POS_X_1.value], df.iloc[0, DataVarIndex.GATE_POS_Y_1.value], df.iloc[0, DataVarIndex.GATE_POS_Z_1.value]]),
            np.array([df.iloc[0, DataVarIndex.GATE_POS_X_2.value], df.iloc[0, DataVarIndex.GATE_POS_Y_2.value], df.iloc[0, DataVarIndex.GATE_POS_Z_2.value]]),
            np.array([df.iloc[0, DataVarIndex.GATE_POS_X_3.value], df.iloc[0, DataVarIndex.GATE_POS_Y_3.value], df.iloc[0, DataVarIndex.GATE_POS_Z_3.value]]),
            np.array([df.iloc[0, DataVarIndex.GATE_POS_X_4.value], df.iloc[0, DataVarIndex.GATE_POS_Y_4.value], df.iloc[0, DataVarIndex.GATE_POS_Z_4.value]]),
        ]

        gate_yaws = [
            df.iloc[0, DataVarIndex.GATE_Y_1.value],
            df.iloc[0, DataVarIndex.GATE_Y_2.value],
            df.iloc[0, DataVarIndex.GATE_Y_3.value],
            df.iloc[0, DataVarIndex.GATE_Y_4.value],
        ]

        def get_square_corners(center, yaw, size=0.4):
            half = size / 2
            corners = np.array([
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
                [-half, -half]
            ])
            rot = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            rotated = corners @ rot.T
            return rotated + center[:2]  # Only XY

        def get_vertical_square(center, yaw, size=0.4):
            half = size / 2
            # Define corners in the gate's local frame (Y forward, Z up)
            local_corners = np.array([
                [-half, -half],  # left-bottom
                [ half, -half],  # right-bottom
                [ half,  half],  # right-top
                [-half,  half],  # left-top
                [-half, -half],  # close the loop
            ])
            # Rotate around Z (affects X and Y)
            rot = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)],
            ])
            # Apply rotation to Y direction, Z stays
            rotated_xy = local_corners[:, 0:1] * rot[0] + local_corners[:, 1:1+1] * np.array([0, 0]) + center[0:2]  # just shift X and Y
            x = center[0] + local_corners[:, 0] * np.cos(yaw)
            y = center[1] + local_corners[:, 0] * np.sin(yaw)
            z = center[2] + local_corners[:, 1]
            return x, y, z

        def plot_gate_xy(ax, center, yaw):
            x, y, _ = get_vertical_square(center, yaw)
            ax.plot(x, y, 'k-', linewidth=2)

        def plot_gate_xz(ax, center, yaw):
            x, _, z = get_vertical_square(center, yaw)
            ax.plot(x, z, 'k-', linewidth=2)

        def plot_gate_yz(ax, center, yaw):
            _, y, z = get_vertical_square(center, yaw)
            ax.plot(y, z, 'k-', linewidth=2)

        def plot_gate_3d(ax, center, yaw):
            x, y, z = get_vertical_square(center, yaw)
            ax.plot3D(x, y, z, 'k-')#, linewidth=2)


        #fig = plt.figure(figsize=(16, 9))
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4)

        ## XY Plane
        ax = fig.add_subplot(2, 2, 1)
        ax.set_title("XY")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_X.value],
            df.iloc[:, DataVarIndex.DES_POS_Y.value],
            "--", label="ref"
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_X.value],
            df.iloc[:, DataVarIndex.POS_Y.value],
            ":",
            label="meas"
        )
        for pos, yaw in zip(gate_positions, gate_yaws):
            plot_gate_xy(ax, pos, yaw)
        ax.legend()

        ## XZ Plane
        ax = fig.add_subplot(2, 2, 2)
        ax.set_title("XZ")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.axis("equal")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_X.value],
            df.iloc[:, DataVarIndex.DES_POS_Z.value],
            "--", #label="des"
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_X.value],
            df.iloc[:, DataVarIndex.POS_Z.value],
            ":",
            #label="meas"
        )
        for pos, yaw in zip(gate_positions, gate_yaws):
            plot_gate_xz(ax, pos, yaw)
        ax.legend()

        ## YZ Plane
        ax = fig.add_subplot(2, 2, 3)
        ax.set_title("YZ")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("Z [m]")
        ax.axis("equal")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_Y.value],
            df.iloc[:, DataVarIndex.DES_POS_Z.value],
            "--", #label="des"
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_Y.value],
            df.iloc[:, DataVarIndex.POS_Z.value],
            ":",
            #label="meas"
        )
        for pos, yaw in zip(gate_positions, gate_yaws):
            plot_gate_yz(ax, pos, yaw)
        ax.legend()

        ## 3D Plot
        ax = fig.add_subplot(2, 2, 4, projection="3d")
        ax.set_title("3D")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.plot(
            df.iloc[:, DataVarIndex.DES_POS_X.value],
            df.iloc[:, DataVarIndex.DES_POS_Y.value],
            df.iloc[:, DataVarIndex.DES_POS_Z.value],
            "--", #label="des"
        )
        ax.plot(
            df.iloc[:, DataVarIndex.POS_X.value],
            df.iloc[:, DataVarIndex.POS_Y.value],
            df.iloc[:, DataVarIndex.POS_Z.value],
            ":",
            #label="meas"
        )
        for pos, yaw in zip(gate_positions, gate_yaws):
            plot_gate_3d(ax, pos, yaw)

        set_axes_equal(ax)
        #ax.legend()

        
        #ax.set_xlim(x_min, x_max)
        #ax.set_ylim(y_min, y_max)
        #ax.set_zlim(0, 2.0)
        
        #ax.set_box_aspect([1, 1, 1])  # maintain equal scaling
        #ax.view_init(elev=45, azim=-60)  # Better angle


        return fig


    def plot_velocity(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.16, 3))  # IEEE width, good height
        fig.subplots_adjust(hspace=0.3)  # spacing between plots
    
        formatter = ticker.FormatStrFormatter("%.2f")  # 2 digits after decimal

        ax = axs[0]
        ax.set_title("Velocity")
        #ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$v$ [m/s]")
        axs[0].yaxis.set_major_formatter(formatter)

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.VEL_X.value], label=r"$v_{x}$"
        )
        #ax.plot(
        #    timesteps,
        #    df.iloc[:, DataVarIndex.DES_VEL_X.value],
        #    "--",
        #    label=r"$v_{x, des}$",
        #)

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.VEL_Y.value], label=r"$v_{y}$"
        )
        #ax.plot(
        #    timesteps,
        #    df.iloc[:, DataVarIndex.DES_VEL_Y.value],
        #    "--",
        #    label=r"$v_{y, des}$",
        #)

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.VEL_Z.value], label=r"$v_{z}$"
        )
        #ax.plot(
        #    timesteps,
        #    df.iloc[:, DataVarIndex.DES_VEL_Z.value],
        #    #"--",
        #    label=r"$v_{z, des}$",
        #)
        ax.legend()

        ax = axs[1]
        #ax.set_title("Total Velocity")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$|v|$ [m/s]")
        axs[1].yaxis.set_major_formatter(formatter)

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps,
            np.sqrt(
                df.iloc[:, DataVarIndex.VEL_X.value] ** 2
                + df.iloc[:, DataVarIndex.VEL_Y.value] ** 2
                + df.iloc[:, DataVarIndex.VEL_Z.value] ** 2
            ),
            #label=r"$|v| [m/s]$",
        )
        #ax.legend()

        return fig

    def plot_rpy(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        #fig = plt.figure(figsize=(16, 9))
        #fig = plt.figure()

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.16, 4))  # IEEE width, good height
        fig.subplots_adjust(hspace=0.3)  # spacing between plots
    
        formatter = ticker.FormatStrFormatter("%.2f")  # 2 digits after decimal

        ## rpy
        ax = axs[0]
        #ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Angles (RPY)")
        #ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\boldsymbol{\phi}$ [rad]")
        ax.yaxis.set_major_formatter(formatter)

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL.value], label=r"$\phi$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH.value], label=r"$\theta$")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW.value], label=r"$\psi$")

        ax.legend(loc="upper left")

        ## rpy rate
        #ax = fig.add_subplot(2, 1, 2)
        ax = axs[1]
        #ax.set_title("Angle Rates (RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\dot{\boldsymbol{\phi}}$ [rad/s]")
        ax.yaxis.set_major_formatter(formatter)

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.ROLL_RATE.value],
            label=r"$\dot{\phi}$",
        )
        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.PITCH_RATE.value],
            label=r"$\dot{\theta}$",
        )
        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.YAW_RATE.value], 
            label=r"$\dot{\psi}$",
        )

        ax.legend(loc="lower right")

        return fig

    def plot_rpy_raw(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        #fig = plt.figure(figsize=(16, 9))
        fig = plt.figure()


        ## rpy
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Angle Vicon(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL_RAW.value], "-", label=r"roll")
        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.PITCH_RAW.value], "-", label=r"pitch"
        )
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW_RAW.value], "-", label=r"roll")

        ax.legend()

        ## rpy rate
        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Angle Estimator(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL.value], "-", label=r"roll")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH.value], "-", label=r"pitch")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW.value], "-", label=r"yaw")

        ax.legend()

        return fig

    def plot_rpy_raw_each_segment(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ## rpy
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Angle Comparison(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.ROLL_RAW.value], "-", label=r"roll_raw"
        )
        ax.plot(timesteps, df.iloc[:, DataVarIndex.ROLL.value], "-", label=r"roll")
        ax.legend()

        ax = fig.add_subplot(3, 1, 2)
        ax.set_title("Angle Comparison(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.PITCH_RAW.value], "-", label=r"pitch_raw"
        )
        ax.plot(timesteps, df.iloc[:, DataVarIndex.PITCH.value], "-", label=r"pitch")
        ax.legend()

        ax = fig.add_subplot(3, 1, 3)
        ax.set_title("Angle Comparison(RPY)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Angle [rad]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.YAW_RAW.value], "-", label=r"yaw_raw"
        )
        ax.plot(timesteps, df.iloc[:, DataVarIndex.YAW.value], "-", label=r"yaw")
        ax.legend()

        return fig

    def plot_position_raw(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ## rpy
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Position Estimator(m)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"[m]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_X.value], "-", label=r"x [m]")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_Y.value], "-", label=r"y [m]")
        ax.plot(timesteps, df.iloc[:, DataVarIndex.POS_Z.value], "-", label=r"z [m]")

        ax.legend()

        ## rpy rate
        ax = fig.add_subplot(3, 1, 2)
        ax.set_title("Position Vicon(m)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"[m]")

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.POS_RAW_X.value], "-", label=r"x [m]"
        )
        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.POS_RAW_Y.value], "-", label=r"y [m]"
        )
        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.POS_RAW_Z.value], "-", label=r"z [m]"
        )

        ax.legend()

        ax = fig.add_subplot(3, 1, 3)
        ax.set_title("Error (m)")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"[m]")

        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.POS_X.value]
            - df.iloc[:, DataVarIndex.POS_RAW_X.value],
            "-",
            label=r"x [m]",
        )
        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.POS_Y.value]
            - df.iloc[:, DataVarIndex.POS_RAW_Y.value],
            "-",
            label=r"y [m]",
        )
        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.POS_Z.value]
            - df.iloc[:, DataVarIndex.POS_RAW_Z.value],
            "-",
            label=r"z [m]",
        )

        ax.legend()

        return fig

    def plot_attitude_input(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        formatter = ticker.FormatStrFormatter("%.2f")  # 2 digits after decimal

        # check if attitude interface was actually used
        if df.iloc[1, DataVarIndex.CTRL_MODE] != "ATTITUDE":
            print(f"Skipping Attitude input plot, no attitude interface used.")
            return None

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.16, 3))
        fig.subplots_adjust(hspace=0.3)  # spacing between plots

        
        #ax = fig.add_subplot(2, 1, 1)
        ax = axs[0]
        #ax.set_title("Control inputs generated by the MPC during the race")
        ax.set_title("Attitude Interface Inputs")
        #ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$f_{T, cmd}$ [N]")
        ax.yaxis.set_major_formatter(formatter)

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps, df.iloc[:, DataVarIndex.DES_THRUST], "-", #label=r"$f_{T, cmd}$"
        )
        #ax.legend()

        #ax = fig.add_subplot(2, 1, 2)
        ax = axs[1]
        #ax.set_title("Angle Commands")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\boldsymbol{\phi}_{cmd}$ [rad]")
        ax.yaxis.set_major_formatter(formatter)

        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.DES_ROLL.value],
            #"-",
            label=r"$\phi_{cmd}$", #[rad]",
        )
        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.DES_PITCH.value],
            #"-",
            label=r"$\theta_{cmd}$",# [rad]",
        )
        #ax.plot(
        #    timesteps, df.iloc[:, DataVarIndex.DES_YAW.value], "-", #label=r"$yaw_{des}$"
        #)

        ax.legend()

        return fig

    def plot_s_v_s(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        formatter = ticker.FormatStrFormatter("%.2f")  # 2 digits after decimal

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.16, 3))
        fig.subplots_adjust(hspace=0.3)  # spacing between plots

        #fig = plt.figure()

        #ax = fig.add_subplot(2, 1, 1)
        ax = axs[0]
        ax.set_title("Path Progress")
        
        #ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$s$ [m]")
        ax.yaxis.set_major_formatter(formatter)

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.THETA])#, "-", label=r"$theta$")
        #ax.legend()

        ax = axs[1]
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$v_{s}$ [m/s]")
        ax.yaxis.set_major_formatter(formatter)

        ax.plot(timesteps, df.iloc[:, DataVarIndex.V_THETA])#, "-", label=r"$v_{s}$")
        #ax.legend()

        return fig

    def plot_opp_predictions(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_title("Opponent 3d Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        ax.plot(
            df.iloc[:, DataVarIndex.OPP_POS_X.value],
            df.iloc[:, DataVarIndex.OPP_POS_Y.value],
            df.iloc[:, DataVarIndex.OPP_POS_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_29)])
        pred_len = 0
        #print(first_row[0, :])
        #print(f"a")
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_01) + pred_len])
        pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Y_01):int(DataVarIndex.OPP_PRED_HORZ_Y_01) + pred_len])
        pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Z_01):int(DataVarIndex.OPP_PRED_HORZ_Z_01) + pred_len])

        for i in range(0, len(pred_x), 100):
            ax.plot(
                pred_x[i, :],
                pred_y[i, :],
                pred_z[i, :],
                label="pred",
            )
        set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_opp_predictions_XY(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Opponent XY Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        #ax.set_zlabel("Z [m]")

        ax.plot(
            df.iloc[:, DataVarIndex.OPP_POS_X.value],
            df.iloc[:, DataVarIndex.OPP_POS_Y.value],
            #df.iloc[:, DataVarIndex.OPP_POS_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_29)])
        pred_len = 0
        #print(first_row[0, :])
        #print(f"a")
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_01) + pred_len])
        pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Y_01):int(DataVarIndex.OPP_PRED_HORZ_Y_01) + pred_len])
        #pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Z_01):int(DataVarIndex.OPP_PRED_HORZ_Z_01) + pred_len])

        for i in range(0, len(pred_x), 100):
            ax.plot(
                pred_x[i, :],
                pred_y[i, :],
                #pred_z[i, :],
                label="pred",
            )
        #set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_opp_predictions_XZ(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Opponent XZ Trajectory")
        ax.set_xlabel("X [m]")
        #ax.set_ylabel("Y [m]")
        ax.set_ylabel("Z [m]")

        ax.plot(
            df.iloc[:, DataVarIndex.OPP_POS_X.value],
            #df.iloc[:, DataVarIndex.OPP_POS_Y.value],
            df.iloc[:, DataVarIndex.OPP_POS_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_29)])
        pred_len = 0
        #print(first_row[0, :])
        #print(f"a")
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_01) + pred_len])
        #pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Y_01):int(DataVarIndex.OPP_PRED_HORZ_Y_01) + pred_len])
        pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Z_01):int(DataVarIndex.OPP_PRED_HORZ_Z_01) + pred_len])

        for i in range(0, len(pred_x), 100):
            ax.plot(
                pred_x[i, :],
                #pred_y[i, :],
                pred_z[i, :],
                label="pred",
            )
        #set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_opp_predictions_YZ(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Opponent YZ Trajectory")
        #ax.set_xlabel("X [m]")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("Z [m]")

        ax.plot(
            #df.iloc[:, DataVarIndex.OPP_POS_X.value],
            df.iloc[:, DataVarIndex.OPP_POS_Y.value],
            df.iloc[:, DataVarIndex.OPP_POS_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_29)])
        pred_len = 0
        #print(first_row[0, :])
        #print(f"a")
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        #pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_01) + pred_len])
        pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Y_01):int(DataVarIndex.OPP_PRED_HORZ_Y_01) + pred_len])
        pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Z_01):int(DataVarIndex.OPP_PRED_HORZ_Z_01) + pred_len])

        for i in range(0, len(pred_y), 100):
            ax.plot(
                #pred_x[i, :],
                pred_y[i, :],
                pred_z[i, :],
                label="pred",
            )
        #set_axes_equal(ax)

        ax.legend()

        return fig


    def plot_opp_vel_predictions_XY(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Opponent Velocity XY Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        #ax.set_zlabel("Z [m]")

        ax.plot(
            df.iloc[:, DataVarIndex.OPP_VEL_X.value],
            df.iloc[:, DataVarIndex.OPP_VEL_Y.value],
            #df.iloc[:, DataVarIndex.OPP_POS_Z.value],
            label="meas",
        )
        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_29)])
        pred_len = 0
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
                
        pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01) + pred_len])
        pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01) + pred_len])
        #pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Z_01):int(DataVarIndex.OPP_PRED_HORZ_Z_01) + pred_len])

        for i in range(0, len(pred_x), 100):
            ax.plot(
                pred_x[i, :],
                pred_y[i, :],
                #pred_z[i, :],
                label="pred",
            )
        #set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_opp_vel_predictions_XZ(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Opponent Velocity XZ Trajectory")
        ax.set_xlabel("X [m]")
        #ax.set_ylabel("Y [m]")
        ax.set_ylabel("Z [m]")

        ax.plot(
            df.iloc[:, DataVarIndex.OPP_VEL_X.value],
            #df.iloc[:, DataVarIndex.OPP_POS_Y.value],
            df.iloc[:, DataVarIndex.OPP_VEL_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_29)])
        pred_len = 0
        #print(first_row[0, :])
        #print(f"a")
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01) + pred_len])
        #pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Y_01):int(DataVarIndex.OPP_PRED_HORZ_Y_01) + pred_len])
        pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01) + pred_len])

        for i in range(0, len(pred_x), 100):
            ax.plot(
                pred_x[i, :],
                #pred_y[i, :],
                pred_z[i, :],
                label="pred",
            )
        #set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_opp_vel_predictions_YZ(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Opponent Velocity YZ Trajectory")
        #ax.set_xlabel("X [m]")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("Z [m]")

        ax.plot(
            #df.iloc[:, DataVarIndex.OPP_POS_X.value],
            df.iloc[:, DataVarIndex.OPP_VEL_Y.value],
            df.iloc[:, DataVarIndex.OPP_VEL_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_29)])
        pred_len = 0

        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        #pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_01) + pred_len])
        pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01) + pred_len])
        pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01) + pred_len])

        for i in range(0, len(pred_y), 100):
            ax.plot(
                #pred_x[i, :],
                pred_y[i, :],
                pred_z[i, :],
                label="pred",
            )
        #set_axes_equal(ax)

        ax.legend()

        return fig


    def plot_opp_vel_predictions(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_title("Opponent 3d Vel Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        ax.plot(
            df.iloc[:, DataVarIndex.OPP_VEL_X.value],
            df.iloc[:, DataVarIndex.OPP_VEL_Y.value],
            df.iloc[:, DataVarIndex.OPP_VEL_Z.value],
            label="meas",
        )

        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_29)])
        pred_len = 0
        #print(first_row)
        #print(first_row[0, :])
        #print(f"a")
        for i in range(28):
            if np.isnan(first_row[i]):#.isnan():
                pred_len = i
                break
        #print(f"pred len: {pred_len}")
                
        vel_pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01) + pred_len])
        vel_pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01) + pred_len])
        vel_pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01) + pred_len])

        for i in range(0, len(vel_pred_x), 100):
            ax.plot(
                vel_pred_x[i, :],
                vel_pred_y[i, :],
                vel_pred_z[i, :],
                label="pred",
            )
        set_axes_equal(ax)

        ax.legend()

        return fig

    def plot_total_thrust_internal(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.16, 3))  # IEEE width, good height
        fig.subplots_adjust(hspace=0.3)  # spacing between plots
    
        formatter = ticker.FormatStrFormatter("%.2f")  # 2 digits after decimal

        ax = axs[0]
        ax.set_title("Cumulative Thrust Approximation")
        #ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"${f}_{T}$ [N]")
        axs[0].yaxis.set_major_formatter(formatter)

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.F_COLLECTIVE],
            #label=r"$integrated collective thrust$",
        )

        ax = axs[1]
        #ax.set_title("v_theta")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$f_{T, cmd}$ [N]")
        axs[0].yaxis.set_major_formatter(formatter)

        ax.plot(
            timesteps,
            df.iloc[:, DataVarIndex.DES_THRUST],
        )
        return fig

    def plot_status(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Solver Status")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$Solver Status$")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.plot(timesteps, df.iloc[:, DataVarIndex.STATUS], "-", label=r"$status$")
        ax.legend()

        return fig

    def plot_compute_time(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)

        fig = plt.figure(figsize=(7.16, 3))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Computation Time")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"Computation Time [ms]")

        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        ax.scatter(
            timesteps,
            df.iloc[:, DataVarIndex.COMPUTE_TIME]*1_000.0,
            # "-",
            alpha=0.2,
            label=r"$Comp. Time$",
        )
        #ax.legend()

        #ax = fig.add_subplot(2, 1, 2)
        #ax.set_title("Computation Time")
        #ax.set_xlabel(r"$t$ [s]")
        #ax.set_ylabel(r"$Computation Time [s]$")
        #ax.set_ylim(0, 0.03)

        #timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

        #ax.scatter(
        #    timesteps,
        #    df.iloc[:, DataVarIndex.COMPUTE_TIME],
        #    # "-",
        #    label=r"$Comp. Time$",
        #)
        #ax.legend()

        return fig

    #def plot_internal_mpc_inputs(self):
    #    # Read the data from the csv file
    #    df = pd.read_csv(self.file_path)

    #    fig = plt.figure()

    #    ax = fig.add_subplot(4, 1, 1)
    #    #ax.set_title("Control Input Derivatives Computed by the MPC Over the Race Duration")
    #    ax.set_xlabel(r"$t$ [s]")
    #    ax.set_ylabel(r"$\dot{f}_{cmd}$ [N/s]")

    #    timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]

    #    ax.plot(
    #        timesteps,
    #        df.iloc[:, DataVarIndex.DELTA_F_COLLECTIVE_CMD],
    #        "-",
    #    )

    #    ax = fig.add_subplot(4, 1, 2)
    #    ax.set_xlabel(r"$t$ [s]")
    #    ax.set_ylabel(r"$\dot{\phi}_{cmd}$ [r/s]")

    #    ax.plot(
    #        timesteps,
    #        df.iloc[:, DataVarIndex.DELTA_R_CMD],
    #        "-",
    #    )

    #    ax = fig.add_subplot(4, 1, 3)
    #    ax.set_xlabel(r"$t$ [s]")
    #    ax.set_ylabel(r"$\dot{\theta}_{cmd}$ [r/s]")

    #    ax.plot(
    #        timesteps,
    #        df.iloc[:, DataVarIndex.DELTA_P_CMD],
    #        "-",
    #    )

    #    #ax = fig.add_subplot(5, 1, 4)
    #    #ax.set_title("Delta Yaw Cmd")
    #    #ax.set_xlabel(r"$t$ [s]")
    #    #ax.set_ylabel(r"$Delta Yaw Cmd [r/s]$")

    #    #ax.plot(
    #    #    timesteps,
    #    #    df.iloc[:, DataVarIndex.DELTA_Y_CMD],
    #    #    "-",
    #    #    label=r"$Delta Yaw cmd$",
    #    #)

    #    ax = fig.add_subplot(4, 1, 4)
    #    ax.set_xlabel(r"$t$ [s]")
    #    ax.set_ylabel(r"$\dot{v}_{\theta}$ [m/s$^2$]")


    #    ax.plot(
    #        timesteps,
    #        df.iloc[:, DataVarIndex.DELTA_V_THETA],
    #        "-",
    #    )

    #    return fig

    def plot_internal_mpc_inputs(self):
        import matplotlib.pyplot as plt
    
        df = pd.read_csv(self.file_path)
    
        timesteps = df.iloc[:, DataVarIndex.VEL_X.TIME]
    
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(7.16, 6))  # IEEE width, good height
        fig.subplots_adjust(hspace=0.3)  # spacing between plots
    
        formatter = ticker.FormatStrFormatter("%.2f")  # 2 digits after decimal
    
        axs[0].set_title("MPC computed Inputs")
    
        axs[0].set_ylabel(r"$\dot{f}_{cmd}$ [N/s]")
        axs[0].yaxis.set_major_formatter(formatter)
        axs[0].plot(timesteps, df.iloc[:, DataVarIndex.DELTA_F_COLLECTIVE_CMD], "-")
    
        axs[1].set_ylabel(r"$\dot{\phi}_{cmd}$ [rad/s]")
        axs[1].yaxis.set_major_formatter(formatter)
        axs[1].plot(timesteps, df.iloc[:, DataVarIndex.DELTA_R_CMD], "-")
    
        axs[2].set_ylabel(r"$\dot{\theta}_{cmd}$ [rad/s]")
        axs[2].yaxis.set_major_formatter(formatter)
        axs[2].plot(timesteps, df.iloc[:, DataVarIndex.DELTA_P_CMD], "-")
    
        axs[3].set_ylabel(r"$\dot{v}_{s}$ [m/s$^2$]")
        axs[3].yaxis.set_major_formatter(formatter)
        axs[3].set_xlabel(r"$t$ [s]")
        axs[3].plot(timesteps, df.iloc[:, DataVarIndex.DELTA_V_THETA], "-")
    
        return fig


    def plot_data_single_it(
        self,
        file_path,
        show=False,
        save=False,
        save_path="plots/",
        create_dir_form_filename=True,
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

        figs.append(self.plot_position_with_gates())
        plot_names.append("position_with_gates")

        figs.append(self.plot_velocity())
        plot_names.append("velocity")

        figs.append(self.plot_rpy())
        plot_names.append("rpy")

        figs.append(self.plot_position_raw())
        plot_names.append("position_raw")

        figs.append(self.plot_rpy_raw())
        plot_names.append("rpy_raw")

        figs.append(self.plot_rpy_raw_each_segment())
        plot_names.append("rpy_raw_segment")

        fig = figs.append(self.plot_s_v_s())
        plot_names.append("path_progress")

        fig = figs.append(self.plot_status())
        plot_names.append("status")

        fig = figs.append(self.plot_internal_mpc_inputs())
        plot_names.append("internal_mpc_inputs")

        fig = figs.append(self.plot_compute_time())
        plot_names.append("Computation_time")

        fig = figs.append(self.plot_total_thrust_internal())
        plot_names.append("Total_thrust_internal")

        ## OPP POSITION
        fig = figs.append(self.plot_opp_predictions())
        plot_names.append("Opp Predictions 3D")

        fig = figs.append(self.plot_opp_predictions_XY())
        plot_names.append("Opp Predictions XY")

        fig = figs.append(self.plot_opp_predictions_XZ())
        plot_names.append("Opp Predictions XZ")

        fig = figs.append(self.plot_opp_predictions_YZ())
        plot_names.append("Opp Predictions YZ")

        ## OPP VELOCITY
        fig = figs.append(self.plot_opp_vel_predictions())
        plot_names.append("Opp Vel Predictions")

        fig = figs.append(self.plot_opp_vel_predictions_XY())
        plot_names.append("Opp Vel Predictions XY")

        fig = figs.append(self.plot_opp_vel_predictions_XZ())
        plot_names.append("Opp Vel Predictions XZ")

        fig = figs.append(self.plot_opp_vel_predictions_YZ())
        plot_names.append("Opp Vel Predictions YZ")


        # only plot if attitude interface selected.
        fig = self.plot_attitude_input()
        if fig:
            figs.append(fig)
            plot_names.append("attitude_input")

        if self.save:
            [self.save_fig(fig, plot_name) for fig, plot_name in zip(figs, plot_names)]

        if self.show:
            plt.show()

    def calculate_pred_error(self):
        # Read the data from the csv file
        df = pd.read_csv(self.file_path)


        # should have shape (N, 3)
        opp_pos = np.array(df.iloc[:, [DataVarIndex.OPP_POS_X.value, DataVarIndex.OPP_POS_Y, DataVarIndex.OPP_POS_Z]])

        # first estimate pred. horizon length
        first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_29)])
        pred_len = 0
        for i in range(28):
            if np.isnan(first_row[i]):
                pred_len = i
                break

        pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_X_01):int(DataVarIndex.OPP_PRED_HORZ_X_01) + pred_len])
        pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Y_01):int(DataVarIndex.OPP_PRED_HORZ_Y_01) + pred_len])
        pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_Z_01):int(DataVarIndex.OPP_PRED_HORZ_Z_01) + pred_len])
        # Create pred array of size (N, 25, 3)
        pred = np.stack((pred_x, pred_y, pred_z), axis=2)

        print(f"shape pred: {np.shape(pred)}")
        print(f"shape opp pos: {np.shape(opp_pos)}")

        ctrl_freq = df.iloc[1, DataVarIndex.CTRL_FREQ]
        pred_freq = df.iloc[1, DataVarIndex.PRED_FREQ]

        print(f"ctrl freq: {ctrl_freq}")
        print(f"pred freq: {pred_freq}")

        # assert that one number is divisible by the other.
        assert (ctrl_freq % pred_freq == 0)
        sampling_factor = int(ctrl_freq/pred_freq)

        opp_pos_resampled = opp_pos[::sampling_factor, :]
        pred_resampled = pred[::sampling_factor, : , :]

        print(f"shape respampled pred: {np.shape(pred_resampled)}")
        print(f"shape resampled opp pos: {np.shape(opp_pos_resampled)}")

        rmse_horz_list = []
        rmse_list = []
        # Go over all the samples
        for i in range(len(opp_pos_resampled)-pred_len):
            #print(i)
            opp_pos_over_horizon = opp_pos_resampled[i:i+pred_len, :]
            pred_over_horizon = pred_resampled[i, :, :]

            e_horiz = opp_pos_over_horizon - pred_over_horizon
            root_squared_error_over_horizon = np.sqrt(np.sum((e_horiz)**2, axis=1))

            rmse = np.mean(root_squared_error_over_horizon)

            rmse_horz_list += [root_squared_error_over_horizon]
            rmse_list += [rmse]

            #print(f"error_over_horizon = {root_squared_error_over_horizon}")
            #print(f"total prediction horizoin rmse: {np.mean(root_squared_error_over_horizon)}")

        print(f"shape avg e horz: {np.shape(np.array(rmse_horz_list))}")
        avg_e_horz = np.mean(np.array(rmse_horz_list), axis=0)

        # Create Figure
        fig0 = plt.figure()
        ax = fig0.add_subplot(1, 1, 1)
        ax.set_title("Prediction Error over Horizon")
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("RMSE")
        ax.plot(avg_e_horz)

        fig1 = plt.figure()
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_title("Prediction Error along Trajectory")
        ax.set_xlabel("Traj step")
        ax.set_ylabel("RMSE")
        ax.plot(rmse_list)

        print(f"toal mean error: {np.mean(rmse_list)}")

        #ax.plot(timesteps, df.iloc[:, DataVarIndex.STATUS], "-", label=r"$status$")
        #ax.legend()

        return [fig0, fig1]


    def plot_data_multi_it(
        self,
        file_path,
        show=False,
        save=False,
        save_path="plots/",
        create_dir_form_filename=True,
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

        figs += self.calculate_pred_error()
        plot_names += ["Prediction Error Horizon", "Prediction Error Trajectory"]


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



#ax.plot(
#    df.iloc[:, DataVarIndex.OPP_VEL_X.value],
#    df.iloc[:, DataVarIndex.OPP_VEL_Y.value],
#    df.iloc[:, DataVarIndex.OPP_VEL_Z.value],
#    label="meas",
#)

#first_row = np.array(df.iloc[0, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_29)])
#pred_len = 0
##print(first_row)
##print(first_row[0, :])
##print(f"a")
#for i in range(28):
#    if np.isnan(first_row[i]):#.isnan():
#        pred_len = i
#        break
##print(f"pred len: {pred_len}")
#        
#vel_pred_x = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_X_01) + pred_len])
#vel_pred_y = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Y_01) + pred_len])
#vel_pred_z = np.array(df.iloc[:, int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01):int(DataVarIndex.OPP_PRED_HORZ_VEL_Z_01) + pred_len])

#for i in range(0, len(vel_pred_x), 100):
#    ax.plot(
#        vel_pred_x[i, :],
#        vel_pred_y[i, :],
#        vel_pred_z[i, :],
#        label="pred",
#    )
#set_axes_equal(ax)

#ax.legend()

#return fig

# Calculate statistics
def get_stats(times, crashes, wins):
    time_mean = np.nanmean(times)
    time_std = np.nanstd(times)
    crash_rate = np.mean(crashes) * 100
    win_rate = np.mean(wins) * 100
    return time_mean, time_std, crash_rate, win_rate



def plot_table_stats():
    repetitions = 100
    controllers = ["pid", "learning"]
    predictors = ["linear", "acados", "learning"]

    for controller in controllers:
        root = Path(__file__).parents[1] / "saves/exp_prediction_error" / controller

        linear_times = []
        linear_crash = []
        linear_win = []

        acados_times = []
        acados_crash = []
        acados_win = []

        untrained_times = []
        untrained_crash = []
        untrained_win = []

        trained_times = []
        trained_crash = []
        trained_win = []

        for predictor in predictors:
            path = root / predictor #[root / "linear", root / "acados", root / "learning"]
            for rep in range(repetitions):
                rep_path = path / f"{rep:.1f}"
                if not rep_path.exists():
                    continue
                print(f"loading path {rep_path}")

                inner_rep_dfs = load_csvs_as_dfs_from_folder(rep_path)

                for i, df in enumerate(inner_rep_dfs):

                    opp_finish_idx = (df["OPP_TARGET_GATE"] == -1).idxmax()
                    finish_idx = (df["TARGET_GATE"] == -1).idxmax()
                    print(f"opp finish time: {opp_finish_idx}, finish idx: {finish_idx}")
                    finish_time = None
                    crash = False
                    if finish_idx == 0 and df["TARGET_GATE"][0] != -1:
                        finish_time = float("nan")  # No -1 found
                        crash = True
                        print(f"not finished")
                    else:
                        #finish_time = df["TIME"][finish_idx]
                        finish_time = finish_idx * 1/df["CTRL_FREQ"][0]
                        print(f"finish time: {finish_time}")
                    winner = not np.isnan(finish_time) and (
                        finish_idx <= opp_finish_idx or opp_finish_idx == 0
                    )

                    #crash = (df["POS_Z"] < 0).any() and not (df["TARGET_GATE"] == -1).any()
                    #crash = not (df["TARGET_GATE"] == -1).any()
                    #print(f"crash: {crash}")
                    if path.name == "acados" and (i==0 or i==len(inner_rep_dfs) - 1):
                        acados_times.append(finish_time)
                        acados_crash.append(crash)
                        acados_win.append(winner)
                    if path.name == "learning" and i == 0:
                        untrained_times.append(finish_time)
                        untrained_crash.append(crash)
                        untrained_win.append(winner)
                    if path.name == "learning" and i == len(inner_rep_dfs) - 1:  # Last result
                        trained_times.append(finish_time)
                        trained_crash.append(crash)
                        trained_win.append(winner)
                    if path.name == "linear" and (i==0 or i==len(inner_rep_dfs) - 1):
                        linear_times.append(finish_time)
                        linear_crash.append(crash)
                        linear_win.append(winner)



        untrained_stats = get_stats(untrained_times, untrained_crash, untrained_win)
        trained_stats = get_stats(trained_times, trained_crash, trained_win)
        linear_stats = get_stats(linear_times, linear_crash, linear_win)
        acados_stats = get_stats(acados_times, acados_crash, acados_win)


        lap_time_data = [
            [x for x in untrained_times if not np.isnan(x)],
            [x for x in trained_times if not np.isnan(x)],
            [x for x in linear_times if not np.isnan(x)],
            [x for x in acados_times if not np.isnan(x)],
        ]


        # Set up the plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        labels = ["Untrained", "Trained", "Linear", "MPC"]
        x = np.arange(len(labels))
        width = 0.35

        print(f"lap time data untrained:{untrained_times}")
        print(f"shape lap time data untrained:{np.shape(untrained_times)}")
        print(f"lap time data trained:{trained_times}")
        print(f"shape lap time data trained :{np.shape(trained_times)}")
        print(f"lap time data linear:{linear_times}")
        print(f"shape lap time data linear:{np.shape(linear_times)}")
        print(f"lap time dataacados:{acados_times}")
        print(f"shape lap time data acados:{np.shape(acados_times)}")
        assert(len(lap_time_data) == len(labels))

        # Create box plot on ax1
        ax1.boxplot(
            lap_time_data,
            labels=labels,
            showmeans=True,
            meanline=True,
            notch=False
        )

        ax1.set_ylabel("Lap Time (s)")
        ax1.set_title("Mean Lap Times")

        # Plot crash rates
        crashes = [untrained_stats[2], trained_stats[2], linear_stats[2], acados_stats[2]]
        ax2.bar(x, crashes, width)
        ax2.set_ylabel("Crash Rate (%)")
        ax2.set_title("Crash Rates")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)

        # Plot win rates
        wins = [untrained_stats[3], trained_stats[3], linear_stats[3], acados_stats[3]]
        print(f"wins: {wins}")
        ax3.bar(x, wins, width)
        ax3.set_ylabel("Win Rate (%)")
        ax3.set_title("Win Rates")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)

        plt.tight_layout()
        save_path = f"summary_plots/table_stats_{controller}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400)
        plt.show()


def plot_all():
    #plot_table_stats()
    predictors = ["acados", "learning", "linear"]
    opp_ctrls = ["pid", "learning"]
    repetitions= np.arange(10.0)
    taus = np.arange(5)
    def plot_data_cli(prefix="sim", save=True, show=False):
        #drone_idxs = np.arange(2)
        #for drone_index in drone_idxs:
        for opp_ctrl in opp_ctrls:
            for predictor in predictors:
                for repetition in repetitions:
                    for tau in taus:
                        try:
                            plotter = Plotter()
                            file_path = (
                                Path(__file__).parent.parent
                                / Path("saves")
                                / Path("exp_prediction_error")
                                / Path(opp_ctrl)
                                / Path(predictor)
                                / Path(str(repetition))
                                / f"run00{int(tau)}.csv"
                            )
                            save_path = Path(__file__).parent / "single" / opp_ctrl / predictor / str(repetition) / f"run00{int(tau)}"
                            print(f"plotting from path {file_path}")
                            print(f"saving to {save_path}")
                            plotter.plot_data_single_it(file_path, show, save, save_path=save_path)
                            plotter.plot_data_multi_it(file_path, show, save, save_path=save_path)
                        except Exception as e:
                            print(f"plotting for index {0} not possible: {e}")
                            # Break if one the data file for one iteration does not exist under the assumption that the other ones also dont exist.
                            break
                        plt.close("all")

    fire.Fire(plot_data_cli)


def plot_one():
    #plot_table_stats()
    n_run = 0
    
    #predictors = ["acados", "learning", "linear"]
    #opp_ctrls = ["pid", "learning"]
    #repetitions= np.arange(10.0)
    #taus = np.arange(5)
    def plot_data_cli(prefix="sim", save=True, show=False):
        #drone_idxs = np.arange(2)
        #for drone_index in drone_idxs:
        try:
            plotter = Plotter()
            file_path = (
                Path(__file__).parent.parent
                / Path("saves")
                / "deploy"
                / f"run00{int(n_run)}.csv"
            )
            save_path = Path(__file__).parent / "single" / "deploy" / f"run00{int(n_run)}"
            print(f"plotting from path {file_path}")
            print(f"saving to {save_path}")
            plotter.plot_data_single_it(file_path, show, save, save_path=save_path)
            plotter.plot_data_multi_it(file_path, show, save, save_path=save_path)
        except Exception as e:
            print(f"plotting for index {0} not possible: {e}")
            # Break if one the data file for one iteration does not exist under the assumption that the other ones also dont exist.
        plt.close("all")

    fire.Fire(plot_data_cli)


if __name__ == "__main__":
    plot_one()