
"""Figure and graph-analysis helpers (Q3, Q5, Q6). Each plot saves a PDF if given a path."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEPS_PER_DAY = 96          # 15-minute intervals
STEPS_PER_WEEK = 7 * STEPS_PER_DAY


def _finish(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Q3
# --------------------------------------------------------------------------
def plot_sensors_one_week(flow, timestamps, sensor_indices, sensor_ids, week_start="2020-03-02",
                          save_path=None):
    """Three sensors over one week (default: Monday 2020-03-02, pre-lockdown)."""
    start = int(np.searchsorted(timestamps, pd.Timestamp(week_start)))
    week_slice = slice(start, start + STEPS_PER_WEEK)
    fig, axes = plt.subplots(len(sensor_indices), 1, figsize=(11, 2.4 * len(sensor_indices)),
                             sharex=True)
    for axis, sensor_index in zip(np.atleast_1d(axes), sensor_indices):
        axis.plot(timestamps[week_slice], flow[week_slice, sensor_index], linewidth=1)
        axis.set_ylabel("veh / 15 min")
        axis.set_title(f"Sensor {sensor_ids[sensor_index]} (index {sensor_index})", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("Time")
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_adjacency_heatmap(adjacency, title="Road-distance adjacency", save_path=None):
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(adjacency, cmap="viridis", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Sensor index")
    axis.set_ylabel("Sensor index")
    fig.colorbar(image, ax=axis)
    return _finish(fig, save_path)


def plot_degree_histogram(adjacency, save_path=None):
    degrees = (adjacency != 0).sum(axis=1)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(degrees, bins=range(int(degrees.max()) + 2), edgecolor="black")
    axis.set_xlabel("Number of neighbours")
    axis.set_ylabel("Number of sensors")
    axis.set_title("Degree distribution")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------
# Q6(a)
# --------------------------------------------------------------------------
def plot_pred_vs_actual(results_dir, variants, sensor_position, horizon_step=0,
                        test_start="2021-04-05", timestamps_test=None, save_path=None):
    """Predicted vs actual for one forecast horizon over one test week.

    results_dir holds {variant}_test_predictions.npz; sensor_position indexes the saved
    sensors. Window i forecasts time (i + 8 + horizon_step) of the test split.
    """
    fig, axis = plt.subplots(figsize=(11, 3.2))
    reference = np.load(Path(results_dir) / f"{variants[0]}_test_predictions.npz")
    target = reference["target"][:, horizon_step, sensor_position]
    window_offset = 8 + horizon_step                       # input length + horizon index
    x_axis = (timestamps_test[window_offset: window_offset + len(target)]
              if timestamps_test is not None else np.arange(len(target)))
    start = (int(np.searchsorted(x_axis, pd.Timestamp(test_start)))
             if timestamps_test is not None else 0)
    week = slice(start, start + STEPS_PER_WEEK)
    axis.plot(x_axis[week], target[week], color="black", linewidth=1.4, label="Actual")
    for variant in variants:
        prediction = np.load(Path(results_dir) / f"{variant}_test_predictions.npz")["prediction"]
        axis.plot(x_axis[week], prediction[:, horizon_step, sensor_position][week],
                  linewidth=0.9, label=variant)
    axis.set_ylabel("veh / 15 min")
    axis.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    return _finish(fig, save_path)


def results_table(results_dir, variants=("temporal", "fixed", "adaptive", "hybrid")):
    """DataFrame for the Q6(a) table (test metrics, training seconds)."""
    adjacency_name = {"temporal": "None", "fixed": "Fixed", "adaptive": "Adaptive",
                      "hybrid": "Hybrid"}
    rows = []
    for variant in variants:
        result = json.loads((Path(results_dir) / f"{variant}.json").read_text())
        rows.append({
            "Model": "Temporal-only" if variant == "temporal" else "STGNN",
            "Adjacency": adjacency_name[variant],
            "Val MAE": round(result["val"]["mae"], 2),
            "Val RMSE": round(result["val"]["rmse"], 2),
            "Test MAE": round(result["test"]["mae"], 2),
            "Test RMSE": round(result["test"]["rmse"], 2),
            "Training time (s)": round(result["training_seconds"], 1),
            "Epochs": result["epochs_run"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q5(b) / Q6(b): learned graph
# --------------------------------------------------------------------------
def plot_learned_adjacency(learned_adjacency, save_path=None):
    return plot_adjacency_heatmap(learned_adjacency, "Learned adaptive adjacency", save_path)


def top_k_edge_set(adjacency, k):
    """Set of (row, col) pairs for the k largest off-diagonal entries per row."""
    scores = adjacency.copy()
    np.fill_diagonal(scores, -np.inf)
    columns = np.argsort(-scores, axis=1)[:, :k]
    return {(row, int(col)) for row in range(scores.shape[0]) for col in columns[row]}


def edge_overlap_with_fixed(learned_adjacency, fixed_adjacency, k=10):
    """Fraction of the strongest learned edges that exist in the road graph (Q6b-1)."""
    learned_edges = top_k_edge_set(learned_adjacency, k)
    road_edges = {(int(i), int(j)) for i, j in zip(*np.nonzero(fixed_adjacency)) if i != j}
    in_road_graph = sum(edge in road_edges for edge in learned_edges)
    return in_road_graph / len(learned_edges)


def haversine_km(lat_1, lng_1, lat_2, lng_2):
    lat_1, lng_1, lat_2, lng_2 = map(np.radians, (lat_1, lng_1, lat_2, lng_2))
    a = (np.sin((lat_2 - lat_1) / 2) ** 2
         + np.cos(lat_1) * np.cos(lat_2) * np.sin((lng_2 - lng_1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def learned_edge_distances_km(learned_adjacency, meta_sd, k=10):
    """Geographic distance of the strongest learned edges (Q6b-2)."""
    latitudes, longitudes = meta_sd["Lat"].to_numpy(), meta_sd["Lng"].to_numpy()
    edges = np.array(sorted(top_k_edge_set(learned_adjacency, k)))
    return haversine_km(latitudes[edges[:, 0]], longitudes[edges[:, 0]],
                        latitudes[edges[:, 1]], longitudes[edges[:, 1]])


def plot_edge_distance_comparison(learned_adjacency, fixed_adjacency, meta_sd, k=10,
                                  save_path=None):
    """Histogram of learned-edge distances vs road-graph-edge distances."""
    learned_km = learned_edge_distances_km(learned_adjacency, meta_sd, k)
    rows, cols = np.nonzero(fixed_adjacency)
    keep = rows != cols
    road_km = haversine_km(meta_sd["Lat"].to_numpy()[rows[keep]], meta_sd["Lng"].to_numpy()[rows[keep]],
                           meta_sd["Lat"].to_numpy()[cols[keep]], meta_sd["Lng"].to_numpy()[cols[keep]])
    fig, axis = plt.subplots(figsize=(6.5, 4))
    axis.hist(road_km, bins=40, alpha=0.6, density=True, label="Road-graph edges")
    axis.hist(learned_km, bins=40, alpha=0.6, density=True, label=f"Top-{k} learned edges")
    axis.set_xlabel("Straight-line distance between sensors (km)")
    axis.set_ylabel("Density")
    axis.legend()
    return _finish(fig, save_path), learned_km, road_km
