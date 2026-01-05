import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ijson
import seaborn as sns


def compute_speed(timestamps, x_positions, y_positions):
    """
    Computes speed in km/h from position and time data.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        x_positions (np.ndarray): Array of x positions in meters.
        y_positions (np.ndarray): Array of y positions in meters.

    Returns:
        np.ndarray: Speed in km/h.
    """
    time_diffs = np.diff(timestamps)
    time_diffs[time_diffs == 0] = np.nan  # Avoid division by zero

    dx = np.diff(x_positions)
    dy = np.diff(y_positions)
    displacements = np.sqrt(dx**2 + dy**2)  # Euclidean distance

    speeds = displacements / time_diffs  # m/s
    speeds *= 3.6  # Convert to km/h

    return np.insert(speeds, 0, np.nan)  # Align size by inserting NaN at the start


def load_trajectories(
    file_path,
    trajectory_timeframe=pd.Timedelta(minutes=10),
    min_time=None,
    direction_str="west",
):
    """
    Loads trajectories from a given file path, filters by direction and time range.

    Args:
        file_path (str): Path to the file containing the trajectories.
        trajectory_timeframe (pd.Timedelta): Time range for which to load trajectories. Default is 10 minutes.
        min_time (pd.Timestamp): Minimum time for which to load trajectories. Default is None.
        direction_str (str): Direction for which to load trajectories. Default is "west".

    Returns:
        pd.DataFrame: DataFrame containing the loaded trajectories, with columns "trajectory_id", "timestamp", "x_position", "y_position", and "speed".
    """
    if direction_str == "west":
        direction_num = -1
    if direction_str == "east":
        direction_num = 1
    westbound_trajectories = []
    t_min = None
    t_max = None
    MIN_MILE_MARKER = 58.8 * 5280 * 0.3048  # meters
    MAX_MILE_MARKER = 62.8 * 5280 * 0.3048  # 2800 meters
    # Open file and stream data
    with open(file_path, "r") as f:
        trajectory_iterator = ijson.items(f, "item")

        for traj in trajectory_iterator:
            # Mile marker 61 is 322080 feet or 98170 m
            # Mile marker 62 is 327360 feet or 99779.3 m
            x_positions = (
                np.array(traj.get("x_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            y_positions = (
                np.array(traj.get("y_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            direction = traj.get("direction")

            if len(x_positions) > 1 and direction == direction_num:
                timestamps = np.array(traj.get("timestamp", []), dtype=np.float64)
                timestamps = (
                    pd.to_datetime(timestamps, unit="s").astype(np.int64) / 1e9
                )  # Convert to seconds

                if min_time and (timestamps[0] < min_time.timestamp()):
                    continue

                westbound_trajectories.append(
                    {
                        "trajectory": traj,
                        "timestamps": timestamps,
                        "x_positions": x_positions,
                        "y_positions": y_positions,
                    }
                )

                # Efficient min/max tracking
                t_min = timestamps[0] if t_min is None else min(t_min, timestamps[0])
                t_max = timestamps[0] if t_max is None else max(t_max, timestamps[0])

                if (
                    t_max is not None
                    and t_min is not None
                    and (t_max - t_min) > trajectory_timeframe.total_seconds()
                ):
                    break

    print(f"Loaded {len(westbound_trajectories)} westbound trajectories.")

    if not westbound_trajectories:
        return pd.DataFrame(
            columns=["trajectory_id", "timestamp", "x_position", "speed"]
        )

    # Vectorized DataFrame creation
    all_trajectory_ids = []
    all_timestamps = []
    all_x_positions = []
    all_y_positions = []

    for idx, traj in enumerate(westbound_trajectories):
        mask = (traj["x_positions"] >= MIN_MILE_MARKER) & (
            traj["x_positions"] <= MAX_MILE_MARKER
        )

        filtered_timestamps = traj["timestamps"][mask]
        filtered_x_positions = traj["x_positions"][mask]
        filtered_y_positions = traj["y_positions"][mask]

        num_points = len(filtered_timestamps)
        all_trajectory_ids.extend([idx] * num_points)
        all_timestamps.extend(filtered_timestamps)
        all_x_positions.extend(filtered_x_positions)
        all_y_positions.extend(filtered_y_positions)
    df = pd.DataFrame(
        {
            "trajectory_id": np.array(all_trajectory_ids, dtype=np.int32),
            "timestamp": pd.to_datetime(all_timestamps, unit="s"),
            "x_position": np.array(all_x_positions, dtype=np.float32),
            "y_position": np.array(all_y_positions, dtype=np.float32),
        }
    )

    print(df.columns.tolist())
    print(df)

    return df


def get_flow_density_matrix(
    df,
    time_interval=pd.Timedelta(minutes=1),
    space_interval=100,
    output_filename="output.csv",
):
    """
    Computes flow and density matrices from a given DataFrame, given as follows:

    Args:
        df (pd.DataFrame): DataFrame containing the trajectories, with columns "trajectory_id", "timestamp", "x_position", "speed".
        time_interval (pd.Timedelta): Time interval for which to compute the flow and density matrices. Default is 1 minute.
        space_interval (int): Space interval for which to compute the flow and density matrices. Default is 100 meters.
        output_filename (str): Output filename for the flow and density matrices. Default is "output.csv".

    Returns:
        np.ndarray: Flow matrix.
        np.ndarray: Density matrix.
    """
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    x_min, x_max = df["x_position"].min(), df["x_position"].max()

    print("xmax", x_max)
    print("x_min", x_min)
    # Ensure valid ranges
    if x_min == x_max:
        raise ValueError(
            "x_min and x_max are identical, meaning no variation in x_position."
        )

    # Create time and space bins
    time_bins = pd.date_range(start=t_min, end=t_max, freq=time_interval)
    space_bins = np.arange(x_min, x_max + space_interval, space_interval)

    if len(space_bins) < 2:
        raise ValueError(
            "space_bins array is empty or too small, adjust space_interval."
        )

    # Assign bin indices using `pd.cut()`
    df["time_bin"] = pd.cut(
        df["timestamp"], bins=time_bins, labels=False, include_lowest=True
    )
    df["space_bin"] = pd.cut(
        df["x_position"], bins=space_bins, labels=False, include_lowest=True
    )

    # Remove NaNs (out-of-range values)
    df = df.dropna(subset=["time_bin", "space_bin"]).astype(
        {"time_bin": int, "space_bin": int}
    )

    # Compute flow and density using `groupby()`
    flow_matrix = np.zeros((len(time_bins) - 1, len(space_bins) - 1))
    density_matrix = np.zeros_like(flow_matrix)
    lane_matrix = np.zeros_like(flow_matrix)

    grouped = df.groupby(["time_bin", "space_bin"])
    area_bin = (
        (space_interval / 1000.0) * time_interval.total_seconds() / 3600.0
    )  # convert space interval to kilometers, time_interval to hours
    for (time_bin, space_bin), group in grouped:
        # print(time_bin, space_bin)
        traj_group = group.groupby("trajectory_id")
        traj_dict = {traj_id: traj_data for traj_id, traj_data in traj_group}

        total_distance = sum(
            traj_group["x_position"].apply(lambda x: x.max() - x.min())
        )
        total_time = sum(
            traj_group["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds())
        )

        flow_matrix[time_bin, space_bin] = (total_distance / (1000.0)) / area_bin
        density_matrix[time_bin, space_bin] = (total_time / (3600.0)) / area_bin
    # Plot histogram of y_position for each space_bin
    space_grouped = df.groupby("space_bin")
    for space_bin, group in space_grouped:
        plt.figure(figsize=(8, 5))
        plt.hist(group["y_position"], bins=30, alpha=0.7)
        plt.title(f"Histogram of y_position for space_bin {space_bin}")
        plt.xlabel("y_position")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"data/histogram_ypos_spacebin_{space_bin}.png")  # Save to file
        plt.close()
    print(grouped)

    full_filepath = "data/" + output_filename
    form_csv(
        flow_matrix,
        density_matrix,
        time_interval,
        space_interval,
        t_min,
        t_max,
        x_min,
        x_max,
        lane_matrix,
        full_filepath,
    )
    return flow_matrix, density_matrix


def form_csv(
    flow_matrix,
    density_matrix,
    time_increment,
    space_increment,
    t_min,
    t_max,
    x_min,
    x_max,
    y_position_ranges,
    output_filename="output.csv",
):
    """
    Save the flow and density matrices to a CSV file.

    Parameters:
    flow_matrix (np.ndarray): Flow matrix.
    density_matrix (np.ndarray): Density matrix.
    time_increment (pd.Timedelta): Time interval.
    space_increment (float): Space interval.
    t_min (pd.Timestamp): Minimum timestamp.
    t_max (pd.Timestamp): Maximum timestamp.
    x_min (float): Minimum x_position.
    x_max (float): Maximum x_position.
    y_position_ranges (np.ndarray): y_position ranges for each space bin.
    output_filename (str): Output filename (default: "output.csv").

    Returns:
    None
    """
    num_time_bins, num_space_bins = flow_matrix.shape

    time_values = np.array([t_min + i * time_increment for i in range(num_time_bins)])
    space_values = np.array(
        [x_min + i * space_increment for i in range(num_space_bins)]
    )

    time_grid, space_grid = np.meshgrid(time_values, space_values, indexing="ij")

    df = pd.DataFrame(
        {
            "Time": time_grid.ravel(),
            "Space": space_grid.ravel(),
            "Flow": flow_matrix.ravel(),
            "Density": density_matrix.ravel(),
            "y_position_range": y_position_ranges.ravel(),
        }
    )

    df.to_csv(output_filename, index=False)

    print(f"CSV file saved as {output_filename}")


def plot_matrices(
    flow_matrix,
    density_matrix,
    time_increment,
    space_increment,
    t_min,
    t_max,
    x_min,
    x_max,
):
    """
    Plot a heatmap of the flow and density matrices, as well as the speed matrix obtained by dividing the flow by the density.

    Parameters:
    flow_matrix (np.ndarray): Flow matrix.
    density_matrix (np.ndarray): Density matrix.
    time_increment (pd.Timedelta): Time interval.
    space_increment (float): Space interval.
    t_min (pd.Timestamp): Minimum timestamp.
    t_max (pd.Timestamp): Maximum timestamp.
    x_min (float): Minimum x_position.
    x_max (float): Maximum x_position.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))

    # Compute reasonable tick marks
    num_time_bins = flow_matrix.shape[0]
    num_space_bins = flow_matrix.shape[1]

    time_ticks = np.linspace(0, num_time_bins - 1, min(10, num_time_bins)).astype(int)
    space_ticks = np.linspace(0, num_space_bins - 1, min(10, num_space_bins)).astype(
        int
    )
    try:
        time_labels = [(t_min + i * time_increment).strftime("%H:%M") for i in time_ticks]
    except:
        time_labels = [(t_min + i * time_increment) for i in time_ticks]
    space_labels = [int(x_min + i * space_increment) for i in space_ticks]

    # Change tick marks to be hours
    time_labels = [f"{int(label/3600)}:00" for label in time_labels]
    
    #Change space tick marks to be in km
    space_labels = [f"{int(label)/1000:.1f}" for label in space_labels]

    #flip y axis ticks
    space_ticks = space_ticks[::-1]

    # Create heatmap for flow matrix
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(
        flow_matrix.T,
        cmap="RdYlGn",
        xticklabels=time_labels,
        yticklabels=space_labels[::-1],
        cbar_kws={"label": "Flow (veh/hr)"},
    )
    plt.title(
        f"Time-Space Diagram of Vehicle Flow\n{time_increment} time increments, {space_increment} meters"
    )
    plt.xlabel("Time")
    plt.ylabel("Space (meters)")
    plt.xticks(time_ticks, time_labels, rotation=45)
    plt.yticks(space_ticks, space_labels)
    # ax1.invert_yaxis()  # Reverse the y-axis

    # Plot the speed matrix
    ax2 = plt.subplot(2, 2, 2)
    sns.heatmap(
        density_matrix.T,
        cmap="RdYlGn",
        xticklabels=time_labels,
        yticklabels=space_labels[::-1],
        cbar_kws={"label": "Average Density (veh/km)"},
    )
    plt.title(
        f"Time-Space Diagram of Vehicle Density\n{time_increment} time increments, {space_increment} meters"
    )
    plt.xlabel("Time")
    plt.ylabel("Space (km)")
    plt.xticks(time_ticks, time_labels, rotation=45)
    plt.yticks(space_ticks, space_labels)
    # ax2.invert_yaxis()  # Reverse the y-axis

    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(
        flow_matrix.T / density_matrix.T,
        cmap="RdYlGn",
        xticklabels=time_labels,
        yticklabels=space_labels[::-1],
        cbar_kws={"label": "Speed (km/h)"},
        vmin=0
    )
    plt.title(
        f"Time-Space Diagram of Vehicle Speed\n{time_increment} time increments, {space_increment} meters"
    )
    plt.xlabel("Time")
    plt.ylabel("Space (meters)")
    plt.xticks(time_ticks, time_labels, rotation=45)
    plt.yticks(space_ticks, space_labels)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
