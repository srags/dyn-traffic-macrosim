import paths
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
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
    t_min = None
    t_max = None
    MIN_MILE_MARKER = 58.8 * 5280 * 0.3048  # meters
    MAX_MILE_MARKER = 62.8 * 5280 * 0.3048  # 2800 meters

    # To keep memory low on multi-hour files, each trajectory is cut to the
    # mile-marker range as soon as it is read, and only these small arrays are kept.
    kept_timestamps = []
    kept_x_positions = []
    kept_y_positions = []
    kept_counts = []  # points kept per trajectory; its index is the trajectory_id

    # Open file and stream data
    with open(file_path, "r") as f:
        # use_float=True returns plain floats instead of Decimal objects, which
        # is about 2.5x faster and gives the same float32/float64 values.
        trajectory_iterator = ijson.items(f, "item", use_float=True)

        for traj in trajectory_iterator:
            # Mile marker 61 is 322080 feet or 98170 m
            # Mile marker 62 is 327360 feet or 99779.3 m
            x_positions = (
                np.array(traj.get("x_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            direction = traj.get("direction")

            if len(x_positions) > 1 and direction == direction_num:
                timestamps = np.array(traj.get("timestamp", []), dtype=np.float64)
                timestamps = (
                    pd.to_datetime(timestamps, unit="s").astype(np.int64) / 1e9
                )  # Convert to seconds

                if min_time and (timestamps[0] < min_time.timestamp()):
                    continue

                y_positions = (
                    np.array(traj.get("y_position", []), dtype=np.float32) * 0.3048
                )  # Convert feet to meters
                mask = (x_positions >= MIN_MILE_MARKER) & (
                    x_positions <= MAX_MILE_MARKER
                )
                kept_timestamps.append(np.asarray(timestamps)[mask])
                kept_x_positions.append(x_positions[mask])
                kept_y_positions.append(y_positions[mask])
                kept_counts.append(int(mask.sum()))

                # Efficient min/max tracking
                t_min = timestamps[0] if t_min is None else min(t_min, timestamps[0])
                t_max = timestamps[0] if t_max is None else max(t_max, timestamps[0])

                if (
                    t_max is not None
                    and t_min is not None
                    and (t_max - t_min) > trajectory_timeframe.total_seconds()
                ):
                    break

    print(f"Loaded {len(kept_counts)} westbound trajectories.")

    if not kept_counts:
        return pd.DataFrame(
            columns=["trajectory_id", "timestamp", "x_position", "speed"]
        )

    # Join the per-trajectory arrays one column at a time, freeing each list as
    # soon as it is joined so two full copies never sit in memory together.
    trajectory_ids = np.repeat(
        np.arange(len(kept_counts), dtype=np.int32), kept_counts
    )
    del kept_counts
    all_timestamps = pd.to_datetime(np.concatenate(kept_timestamps), unit="s")
    del kept_timestamps
    all_x_positions = np.concatenate(kept_x_positions).astype(np.float32, copy=False)
    del kept_x_positions
    all_y_positions = np.concatenate(kept_y_positions).astype(np.float32, copy=False)
    del kept_y_positions

    df = pd.DataFrame(
        {
            "trajectory_id": trajectory_ids,
            "timestamp": all_timestamps,
            "x_position": all_x_positions,
            "y_position": all_y_positions,
        }
    )

    print(df.columns.tolist())
    print(df)

    return df


def _python_sum_by_group(values, groups, num_groups):
    """
    Sums float64 `values` per group, giving exactly what Python's built-in sum()
    gives when run on each group's values in order. Since Python 3.12, sum()
    of floats uses Neumaier compensated summation; earlier versions add plainly.

    Args:
        values (np.ndarray): float64 values, in the order they should be added.
        groups (np.ndarray): group index (0..num_groups-1) of each value, sorted.
        num_groups (int): number of groups.

    Returns:
        np.ndarray: per-group sums (0.0 for empty groups).
    """
    totals = np.zeros(num_groups, dtype=np.float64)
    comp = np.zeros(num_groups, dtype=np.float64)
    if len(values) == 0:
        return totals
    # Position of each value within its group (0 for the first value, ...)
    group_start = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    sizes = np.diff(np.r_[group_start, len(groups)])
    rank = np.arange(len(groups)) - np.repeat(group_start, sizes)

    # Step through the k-th value of every group at once.
    order = np.argsort(rank, kind="stable")
    bounds = np.searchsorted(rank[order], np.arange(rank.max() + 2))
    compensated = sys.version_info >= (3, 12)
    for k in range(rank.max() + 1):
        idx = order[bounds[k] : bounds[k + 1]]
        g, x = groups[idx], values[idx]
        if k == 0:
            totals[g] = x
            continue
        s = totals[g]
        t = s + x
        if compensated:
            comp[g] += np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s)
        totals[g] = t
    if compensated:
        fix = (comp != 0) & np.isfinite(comp)
        totals[fix] += comp[fix]
    return totals


def get_flow_density_matrix(
    df,
    time_interval=pd.Timedelta(minutes=1),
    space_interval=100,
    output_filename="output.csv",
    start_time=None,
    end_time=None,
):
    """
    Computes flow and density matrices from a given DataFrame, given as follows:

    Args:
        df (pd.DataFrame): DataFrame containing the trajectories, with columns "trajectory_id", "timestamp", "x_position", "speed".
        time_interval (pd.Timedelta): Time interval for which to compute the flow and density matrices. Default is 1 minute.
        space_interval (int): Space interval for which to compute the flow and density matrices. Default is 100 meters.
        output_filename (str): Output filename for the flow and density matrices. Default is "output.csv".
        start_time, end_time (pd.Timestamp, optional): Fixed edges of the time grid,
            e.g. 06:00 and 10:00 for exactly 4 hours of bins. If not given, the grid runs
            from the first to the last timestamp in df, and a final partial bin is dropped.

    Returns:
        np.ndarray: Flow matrix.
        np.ndarray: Density matrix.
    """
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    if start_time is not None:
        t_min = pd.Timestamp(start_time)
    if end_time is not None:
        t_max = pd.Timestamp(end_time)
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

    # Keep only points inside the bins (drops NaN bins), as plain numpy arrays
    # so no full copy of the DataFrame is made.
    time_idx = df["time_bin"].to_numpy()
    space_idx = df["space_bin"].to_numpy()
    valid = ~(np.isnan(time_idx) | np.isnan(space_idx))
    time_idx = time_idx[valid].astype(np.int64)
    space_idx = space_idx[valid].astype(np.int64)
    traj_ids = df["trajectory_id"].to_numpy()[valid]
    x_vals = df["x_position"].to_numpy()[valid]  # keeps the column's dtype
    t_ns = df["timestamp"].to_numpy()[valid].astype("datetime64[ns]").view(np.int64)

    num_time_bins, num_space_bins = len(time_bins) - 1, len(space_bins) - 1
    flow_matrix = np.zeros((num_time_bins, num_space_bins))
    density_matrix = np.zeros_like(flow_matrix)
    lane_matrix = np.zeros_like(flow_matrix)

    area_bin = (
        (space_interval / 1000.0) * time_interval.total_seconds() / 3600.0
    )  # convert space interval to kilometers, time_interval to hours

    # Edie's definitions: for each (time bin, space bin), add up each trajectory's
    # distance travelled (max x - min x) and time spent (max t - min t) in that cell.
    # One grouped pass over all points replaces a Python loop over every cell.
    cell = time_idx * num_space_bins + space_idx
    per_traj = (
        pd.DataFrame({"cell": cell, "traj": traj_ids, "x": x_vals, "t": t_ns})
        .groupby(["cell", "traj"], sort=True)
        .agg(x_min=("x", "min"), x_max=("x", "max"), t_min=("t", "min"), t_max=("t", "max"))
    )
    cells = per_traj.index.get_level_values("cell").to_numpy()
    # Subtract in the column's dtype, then add up in float64, as before.
    distances = (per_traj["x_max"] - per_traj["x_min"]).to_numpy().astype(np.float64)
    # Seconds, computed exactly like Timedelta.total_seconds(): truncated to
    # microseconds, then whole seconds + fractional part.
    duration_us = (per_traj["t_max"] - per_traj["t_min"]).to_numpy() // 1000
    durations = (duration_us // 10**6) + (duration_us % 10**6) / 1e6
    del per_traj

    # Add up per cell in trajectory-id order, the same way Python's sum() did in
    # the previous per-cell loop, so the totals match it exactly.
    total_distance = _python_sum_by_group(distances, cells, num_time_bins * num_space_bins)
    total_time = _python_sum_by_group(durations, cells, num_time_bins * num_space_bins)
    occupied = np.unique(cells)
    t_occ, s_occ = np.divmod(occupied, num_space_bins)
    flow_matrix[t_occ, s_occ] = (total_distance[occupied] / (1000.0)) / area_bin
    density_matrix[t_occ, s_occ] = (total_time[occupied] / (3600.0)) / area_bin

    # # Plot histogram of y_position for each space_bin
    # y_vals = df["y_position"].to_numpy()[valid]
    # for space_bin in np.unique(space_idx):
    #     plt.figure(figsize=(8, 5))
    #     plt.hist(y_vals[space_idx == space_bin], bins=30, alpha=0.7)
    #     plt.title(f"Histogram of y_position for space_bin {space_bin}")
    #     plt.xlabel("y_position")
    #     plt.ylabel("Count")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(paths.DATA_ROOT, f"histogram_ypos_spacebin_{space_bin}.png"))  # Save to file
    #     plt.close()

    # full_filepath = os.path.join(paths.DATA_ROOT, output_filename)

    # form_csv(
    #     flow_matrix,
    #     density_matrix,
    #     time_interval,
    #     space_interval,
    #     t_min,
    #     t_max,
    #     x_min,
    #     x_max,
    #     lane_matrix,
    #     full_filepath,
    # )

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
    velocity_matrix,
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
        # t_min is a Timestamp: label as HH:MM
        time_labels = [(t_min + i * time_increment).strftime("%H:%M") for i in time_ticks]
    except AttributeError:
        # t_min is in seconds: label as whole hours
        seconds = [t_min + i * time_increment.total_seconds() for i in time_ticks]
        time_labels = [f"{int(s / 3600)}:00" for s in seconds]
    space_labels = [int(x_min + i * space_increment) for i in space_ticks]

    
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
        velocity_matrix.T,
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
