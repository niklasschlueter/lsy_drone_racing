from pathlib import Path

import os
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_data(path: Path) -> list[pd.DataFrame]:
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


def plot_prediction_3d(timesteps: list[int], actual_positions, predictions, prediction_method=""):
    # Create a new figure for these specific positions
    fig_pos = plt.figure(figsize=(12, 8))
    ax_pos = fig_pos.add_subplot(111, projection="3d")

    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))

    for traj_idx in range(len(actual_positions)):
        N = len(actual_positions)
        alpha = 1.0 / N + (traj_idx * (1.0 - 1.0 / N) / (N - 1))  # Alpha increases from 1/N to 1
        for i, t in enumerate(timesteps):
            if t < actual_positions[traj_idx].shape[0]:
                # Plot the full horizon for this time step
                ax_pos.plot(
                    predictions[traj_idx][t, :, 0],
                    predictions[traj_idx][t, :, 1],
                    predictions[traj_idx][t, :, 2],
                    c=colors[i],
                    alpha=alpha,
                    label=f"Position horizon at t={t}" if traj_idx == 0 else "",
                )
                # Plot the actual trajectory for this horizon in dotted line
                horizon_length = predictions[traj_idx].shape[1]
                if t + horizon_length <= actual_positions[traj_idx].shape[0]:
                    ax_pos.plot(
                        actual_positions[traj_idx][t : t + horizon_length, 0],
                        actual_positions[traj_idx][t : t + horizon_length, 1],
                        actual_positions[traj_idx][t : t + horizon_length, 2],
                        c=colors[i],
                        alpha=alpha,
                        linestyle=":",
                        label=f"Actual trajectory at t={t}" if traj_idx == 0 else "",
                    )
    # Plot full trajectory
    ax_pos.plot(
        actual_positions[traj_idx][:, 0],
        actual_positions[traj_idx][:, 1],
        actual_positions[traj_idx][:, 2],
        linestyle="--",
        alpha=0.2,
    )
    ax_pos.set_xlabel("X Position")
    ax_pos.set_ylabel("Y Position")
    ax_pos.set_zlabel("Z Position")
    ax_pos.set_title(f"Opponent {prediction_method + ' '}Position Estimate at Specific Timesteps")
    ax_pos.legend()

    # Make axes equal
    max_range = (
        np.ptp(np.array([ax_pos.get_xlim(), ax_pos.get_ylim(), ax_pos.get_zlim()]), axis=1).max()
        / 2.0
    )

    mid_x = np.mean(ax_pos.get_xlim())
    mid_y = np.mean(ax_pos.get_ylim())
    mid_z = np.mean(ax_pos.get_zlim())
    ax_pos.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_pos.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_pos.set_zlim(mid_z - max_range, mid_z + max_range)


def compute_mean_prediction_error(actual_positions, predictions):
    n_horizon = predictions.shape[1]
    # Plot actual positions at specific timesteps with full horizon
    # Create array of actual positions shifted to match predictions
    actual_shifted = np.zeros_like(predictions)
    for i in range(len(actual_positions)):
        if i + n_horizon <= len(actual_positions):
            actual_shifted[i] = actual_positions[i : i + n_horizon]
    # Cut off the last n_horizon timesteps since we don't have enough actual positions to compare
    predictions = predictions[: -n_horizon + 1]
    actual_shifted = actual_shifted[: -n_horizon + 1]

    return np.mean(np.linalg.norm(predictions - actual_shifted, axis=-1), axis=-1)


def compute_horizon_error(actual_positions, predictions):
    n_horizon = predictions.shape[1]
    # Plot actual positions at specific timesteps with full horizon
    # Create array of actual positions shifted to match predictions
    actual_shifted = np.zeros_like(predictions)
    for i in range(len(actual_positions)):
        if i + n_horizon <= len(actual_positions):
            actual_shifted[i] = actual_positions[i : i + n_horizon]
    # Cut off the last n_horizon timesteps since we don't have enough actual positions to compare
    predictions = predictions[: -n_horizon + 1]
    actual_shifted = actual_shifted[: -n_horizon + 1]
    return np.mean(np.linalg.norm(predictions - actual_shifted, axis=-1), axis=0)


def plot_joint_error(errors, paths):
    """Create joint error plots comparing all methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Trajectory errors
    # Plot single line for linear and acados
    if "linear" in errors:
        for error in errors["linear"]:
            for e in error:
                ax1.plot(e, color="blue", alpha=1 / len(errors["linear"]))
        ax1.plot([], label="Linear", color="blue")  # Add single label line

    if "acados" in errors:
        for error in errors["acados"]:
            for e in error:
                ax1.plot(e, color="green", alpha=1 / len(errors["acados"]))
        ax1.plot([], label="Acados", color="green")  # Add single label line

    if "learning" in errors:
        for error in errors["learning"]:
            for e in error:
                ax1.plot(e, color="red", alpha=1 / len(errors["learning"]))
        ax1.plot([], label="Learning", color="red")  # Add single label line

        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Total Error (m)")
        ax1.set_title("Prediction Error Over Time")
        ax1.legend()
        ax1.grid(True)

    # Plot 2: Line plot with error bands
    # Plot lines with error bands for linear and acados
    if "linear" in errors:
        linear_means = []
        linear_stds = []
        # Invert the nested lists so inner lists are combined into one outer list
        errors_ep_tau = list(map(list, zip(*errors["linear"])))
        for error in errors_ep_tau:
            e = np.concat(error)
            linear_means.append(np.mean(e))
            linear_stds.append(np.std(e))
        linear_means = np.array(linear_means)
        linear_stds = np.array(linear_stds)
        x_lin = np.arange(len(linear_means))
        ax2.plot(x_lin, linear_means, label="Linear", color="blue")
        ax2.fill_between(
            x_lin, linear_means - linear_stds, linear_means + linear_stds, color="blue", alpha=0.2
        )

    if "acados" in errors:
        acados_means = []
        acados_stds = []
        errors_ep_tau = list(map(list, zip(*errors["acados"])))
        for error in errors_ep_tau:
            e = np.concat(error)
            acados_means.append(np.mean(e))
            acados_stds.append(np.std(e))
        acados_means = np.array(acados_means)
        acados_stds = np.array(acados_stds)

        x_aca = np.arange(len(acados_means))
        ax2.plot(x_aca, acados_means, label="Acados", color="green")
        ax2.fill_between(
            x_aca, acados_means - acados_stds, acados_means + acados_stds, color="green", alpha=0.2
        )

    # Plot learning runs with error bands
    if "learning" in errors:
        learning_means = []
        learning_stds = []
        max_len = max([len(e) for e in errors["learning"]])
        # Only keep max_len errors to prevent errors_ep_tau from truncating
        errors["learning"] = [e for e in errors["learning"] if len(e) == max_len]
        errors_ep_tau = list(map(list, zip(*errors["learning"])))
        for error in errors_ep_tau:
            e = np.concat(error)
            learning_means.append(np.mean(e))
            learning_stds.append(np.std(e))

        learning_means = np.array(learning_means)
        learning_stds = np.array(learning_stds)

        x_learn = np.arange(len(learning_means))
        ax2.plot(x_learn, learning_means, color="red", label="Learning")
        ax2.fill_between(
            x_learn,
            learning_means - learning_stds,
            learning_means + learning_stds,
            color="red",
            alpha=0.2,
        )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean Error (m)")
    ax2.set_title("Average Prediction Error by Method")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_normalized_error(errors, paths):
    """Plot the average prediction error normalized over prediction horizon per run."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "green", "red"]
    labels = ["Linear", "Acados", "Learning"]

    for i, (method_errors, path) in enumerate(zip(errors, paths)):
        # Each element in method_errors is a list of errors for one run
        # We need to normalize each run's errors by the prediction horizon length
        n_horizon = len(method_errors[0])  # Length of prediction horizon
        # Plot each run as a line
        alphas = np.linspace(1, len(errors[i]), len(errors[i])) / len(errors[i])
        for j, run_errors in enumerate(method_errors):
            ax.plot(
                range(n_horizon),
                run_errors,
                color=colors[i],
                alpha=alphas[j],
                linewidth=alphas[j] * 4,
            )

        # Plot mean line
        mean_line = np.mean(method_errors, axis=0)
        ax.plot(range(n_horizon), mean_line, color=colors[i], label=labels[i], linewidth=2)

    ax.set_xlabel("prediction step")
    ax.set_ylabel("Average Error per Prediction Step (m)")
    ax.set_title("Normalized Prediction Error by Method")
    ax.grid(True, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.show()


def run_plots(controller: str = "pid"):
    root = Path(__file__).parents[1] / "saves/exp_prediction_error" / controller
    #paths = [root / "linear", root / "acados", root / "learning"]
    predictors = ["linear", "learning", "acados"]

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
        path = root / predictor
        # Iterate through tau values
        for tau in range(100):
            tau_path = path / f"{tau:.1f}"
            if not tau_path.exists():
                continue

            dfs = load_data(tau_path)

            print(f"tau path: {tau_path}")
            # We need: lap times or NaN
            for i, df in enumerate(dfs):
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
                print(f"winner: {winner}")

                #crash = (df["POS_Z"] < 0).any() and not (df["TARGET_GATE"] == -1).any()
                #crash = not (df["TARGET_GATE"] == -1).any()
                print(f"crash: {crash}")
                if path.name == "acados" and (i == 0 or i==len(dfs) -1):
                    acados_times.append(finish_time)
                    acados_crash.append(crash)
                    acados_win.append(winner)
                if path.name == "learning" and i == 0:
                    untrained_times.append(finish_time)
                    untrained_crash.append(crash)
                    untrained_win.append(winner)
                if path.name == "learning" and i == len(dfs) - 1:  # Last result
                    trained_times.append(finish_time)
                    trained_crash.append(crash)
                    trained_win.append(winner)
                if path.name == "linear" and (i == 0 or i==len(dfs) -1):
                    linear_times.append(finish_time)
                    linear_crash.append(crash)
                    linear_win.append(winner)

    # Calculate statistics
    def get_stats(times, crashes, wins):
        time_mean = np.nanmean(times)
        time_std = np.nanstd(times)
        crash_rate = np.mean(crashes) * 100
        win_rate = np.mean(wins) * 100
        return time_mean, time_std, crash_rate, win_rate

    untrained_stats = get_stats(untrained_times, untrained_crash, untrained_win)
    trained_stats = get_stats(trained_times, trained_crash, trained_win)
    linear_stats = get_stats(linear_times, linear_crash, linear_win)
    acados_stats = get_stats(acados_times, acados_crash, acados_win)

    # Set up the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Untrained", "Trained", "Linear", "MPC"]
    x = np.arange(len(labels))
    width = 0.35

    # Plot lap times with error bars
    #times = [untrained_stats[0], trained_stats[0], linear_stats[0], acados_stats[0]]
    #errors = [untrained_stats[1], trained_stats[1], linear_stats[1], acados_stats[1]]

    # Plot lap times but without nans in the data.
    lap_time_data = [
        [x for x in untrained_times if not np.isnan(x)],
        [x for x in trained_times if not np.isnan(x)],
        [x for x in linear_times if not np.isnan(x)],
        [x for x in acados_times if not np.isnan(x)],
    ]

    print(f"lap time data untrained:{untrained_times}")
    print(f"shape lap time data untrained:{np.shape(untrained_times)}")
    print(f"lap time data trained:{trained_times}")
    print(f"shape lap time data trained :{np.shape(trained_times)}")
    print(f"lap time data linear:{linear_times}")
    print(f"shape lap time data linear:{np.shape(linear_times)}")
    print(f"lap time dataacados:{acados_times}")
    print(f"shape lap time data acados:{np.shape(acados_times)}")
    assert(len(lap_time_data) == len(labels))
    
    import seaborn as sns
    #colors = ["orange", "red", "blue", "green"]
    colors = sns.color_palette("colorblind")
    #sns.set_theme(style="whitegrid")
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    bar_width = 0.3
    ## Create box plot on ax1
    ax1.boxplot(
        lap_time_data,
        labels=labels,
        showmeans=True,
        meanline=True,
        notch=False
    )

    ax1.set_ylabel("Lap Time (s)")
    ax1.set_title("Mean Lap Times")

    ## Plot crash rates
    #crashes = [untrained_stats[2], trained_stats[2], linear_stats[2], acados_stats[2]]
    #ax2.bar(x, crashes, width, color=colors)
    #ax2.set_ylabel("Crash Rate (%)")
    #ax2.set_title("Crash Rates")
    #ax2.set_xticks(x)
    #ax2.set_xticklabels(labels)

    ## Plot win rates
    #wins = [untrained_stats[3], trained_stats[3], linear_stats[3], acados_stats[3]]
    #print(f"wins: {wins}")
    #ax3.bar(x, wins, width, color=colors)
    #ax3.set_ylabel("Win Rate (%)")
    #ax3.set_title("Win Rates")
    #ax3.set_xticks(x)
    #ax3.set_xticklabels(labels)


    crashes = [untrained_stats[2], trained_stats[2], linear_stats[2], acados_stats[2]]
    wins = [untrained_stats[3], trained_stats[3], linear_stats[3], acados_stats[3]]

    # Create figure and axes
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 12))

    # --- Box Plot for Lap Times ---
    # Flatten and label the data for seaborn
    lap_times_flat = [time for group in lap_time_data for time in group]
    methods = [label for label, group in zip(labels, lap_time_data) for _ in group]

    lap_df = pd.DataFrame({
        'Lap Time': lap_times_flat,
        'Method': methods
    })

    #sns.boxplot(x='Method', y='Lap Time', data=lap_df, ax=ax1, palette=colors, showmeans=True,
    #meanprops={"linestyle": "-", "color": "black"})
    #sns.violinplot(data=lap_df, x="Method", y="Lap Time", palette=colors, ax=ax1, inner="quartile")
    #sns.stripplot(data=lap_df, x="Method", y="Lap Time", palette=colors, ax=ax1,jitter=True)


    ax1.set_title("Mean Lap Times")

    # --- Crash Rates Bar Plot ---
    sns.barplot(x=labels, y=crashes, palette=colors, ax=ax2, width=bar_width)
    ax2.set_ylabel("Crash Rate (%)")
    ax2.set_title("Crash Rates")

    # --- Win Rates Bar Plot ---
    sns.barplot(x=labels, y=wins, palette=colors, ax=ax3, width=bar_width)
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_title("Win Rates")

    plt.tight_layout()
    save_path = f"summary_plots/table_stats_{controller}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.show()

def main():
    for controller in ["pid", "learning"]:
        run_plots(controller)


if __name__ == "__main__":
    fire.Fire(main)
