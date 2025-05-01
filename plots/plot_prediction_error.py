from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_data(path: Path) -> list[pd.DataFrame]:
    # Path to the directory containing prediction error data
    # Find the most recent file(s) in the directory
    files = list(path.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {path}")
    # Sort files by modification time (most recent first)
    files.sort(key=lambda x: x.stat().st_mtime)
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    return dfs


def preprocess_data(dfs: list[pd.DataFrame]) -> tuple[list[NDArray], list[NDArray]]:
    actual_positions, predictions = [], []
    for df in dfs:
        # Drop rows where opponent z position is negative -> has been warped out of sim
        df = df[df["OPP_POS_Z"] >= 0]
        # Check that control frequency is constant throughout trajectory
        ctrl_freq_vals = df["CTRL_FREQ"].unique()
        if len(ctrl_freq_vals) != 1:
            raise ValueError(f"Control frequency varies in trajectory: {ctrl_freq_vals}")
        ctrl_freq = ctrl_freq_vals[0]
        # Check that prediction frequency is constant throughout trajectory
        pred_freq_vals = df["PRED_FREQ"].unique()
        if len(pred_freq_vals) != 1:
            raise ValueError(f"Prediction frequency varies in trajectory: {pred_freq_vals}")
        pred_freq = pred_freq_vals[0]
        # Check that control frequency is divisible by prediction frequency
        if not (ctrl_freq % pred_freq == 0):
            raise ValueError(
                f"Control frequency {ctrl_freq} is not divisible by prediction frequency {pred_freq}"
            )
        resample = ctrl_freq // pred_freq

        # Drop columns that contain all NaN values
        df = df.dropna(axis=1, how="all")
        # Find highest horizon index by looking for OPP_PRED_HORZ_X_ columns
        horizon_cols = [col for col in df.columns if col.startswith("OPP_PRED_HORZ_X_")]
        n_horizon = max([int(col.split("_")[-1]) for col in horizon_cols])

        # Create array to store prediction horizon data
        # Horizon data is sampled with mpcc frequency -> check df
        pred_x = np.stack(
            [df[f"OPP_PRED_HORZ_X_{i:02d}"] for i in range(1, n_horizon + 1)], axis=-1
        )
        pred_y = np.stack(
            [df[f"OPP_PRED_HORZ_Y_{i:02d}"] for i in range(1, n_horizon + 1)], axis=-1
        )
        pred_z = np.stack(
            [df[f"OPP_PRED_HORZ_Z_{i:02d}"] for i in range(1, n_horizon + 1)], axis=-1
        )
        pred = np.stack([pred_x, pred_y, pred_z], axis=-1)[::resample]
        predictions.append(pred)

        pos = np.stack([df["OPP_POS_X"], df["OPP_POS_Y"], df["OPP_POS_Z"]], axis=-1)[::resample]
        actual_positions.append(pos)
    return actual_positions, predictions


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


def main(controller: str = "pid"):
    root = Path(__file__).parents[1] / "saves/exp_prediction_error" / controller
    paths = [root / "linear", root / "acados", root / "learning"]

    errors = {}
    horizon_errors = []
    for path in paths:
        errors[path.name] = []
        horizon_errors.append([])

        # Iterate through tau values from 1.0 to 2.0
        for tau in np.linspace(0.0, 1.0, 11):
            tau_path = path / f"{tau:.1f}"
            if not tau_path.exists():
                continue

            dfs = load_data(tau_path)
            actual_positions, predictions = preprocess_data(dfs)

            # Sanity check for predictions
            if path.name == "learning":
                ...
                # # Only plot 3D visualization for tau=1.0 case
                # if tau == 1.0:
                #     plot_prediction_3d(
                #         [30, 60, 100, 130, 160, 200], actual_positions, predictions, path.name
                #     )
                #     plt.show()

            # Compute errors for this tau value
            tau_errors = []
            tau_horizon_errors = []
            for pos, pred in zip(actual_positions, predictions):
                tau_errors.append(compute_mean_prediction_error(pos, pred))
                tau_horizon_errors.append(compute_horizon_error(pos, pred))

            # Add errors from this tau to the overall list
            errors[path.name].append(tau_errors)
            horizon_errors[-1].append(tau_horizon_errors)

    plot_joint_error(errors, paths)


if __name__ == "__main__":
    fire.Fire(main)
