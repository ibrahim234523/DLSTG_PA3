from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SAN_DIEGO_DISTRICT = 11
EXPECTED_NUM_SENSORS = 716
STEPS_PER_15_MIN = 3            # 3 x 5-min samples per 15-min interval
INPUT_LENGTH = 8                # 2 hours of history
HORIZON = 4                     # 1 hour ahead
MAX_FILL_GAP = 4                # forward-fill gaps of up to 4 steps (1 hour)

TRAIN_END = "2020-12-31 23:59"
VAL_START, VAL_END = "2021-01-01 00:00", "2021-03-31 23:59"
TEST_START, TEST_END = "2021-04-01 00:00", "2021-12-31 23:59"


# --------------------------------------------------------------------------
# Q3(a): extraction and resampling
# --------------------------------------------------------------------------
def select_san_diego_sensors(meta_path):
    """Return (meta_sd, sd_row_positions). Row positions index ca_rn_adj.npy."""
    meta = pd.read_csv(meta_path)
    is_san_diego = meta["District"] == SAN_DIEGO_DISTRICT
    sd_row_positions = np.flatnonzero(is_san_diego.to_numpy())
    meta_sd = meta.loc[is_san_diego].reset_index(drop=True)
    assert len(meta_sd) == EXPECTED_NUM_SENSORS, (
        f"Expected {EXPECTED_NUM_SENSORS} SD sensors, found {len(meta_sd)}. "
        "Check the District filter against the LargeST generate_sd_dataset notebook."
    )
    return meta_sd, sd_row_positions


def load_year_flow(h5_path, sensor_ids):
    """Load one year of 5-min flow and keep only the SD sensors, in meta order."""
    raw_flow = pd.read_hdf(h5_path)
    raw_flow.columns = raw_flow.columns.astype(str)
    sd_flow = raw_flow[[str(sensor_id) for sensor_id in sensor_ids]]
    del raw_flow                                  # free memory (CA has ~8600 columns)
    return sd_flow.astype(np.float32)


def resample_to_15_min(flow_5_min):
    """Sum 3 consecutive 5-min counts into one 15-min count.

    LargeST flow is a vehicle count per 5 minutes, so counts are summed (not
    averaged) to keep units as vehicles / 15 min. If one of the three samples is
    missing, the sum of the available two is rescaled by 3/2; with fewer than two
    valid samples the interval is marked NaN (missing).
    """
    grouped = flow_5_min.resample("15min")
    valid_count = grouped.count()
    partial_sum = grouped.sum(min_count=1)
    flow_15_min = partial_sum * (STEPS_PER_15_MIN / valid_count.replace(0, np.nan))
    flow_15_min[valid_count < 2] = np.nan
    return flow_15_min


def build_dataset(raw_dir, processed_dir):
    """Run the full Q3(a)/(b) extraction and cache the result to processed_dir."""
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    meta_sd, sd_row_positions = select_san_diego_sensors(raw_dir / "ca_meta.csv")
    yearly_flows = []
    for year in (2020, 2021):
        flow_5_min = load_year_flow(raw_dir / f"ca_his_raw_{year}.h5", meta_sd["ID"])
        yearly_flows.append(resample_to_15_min(flow_5_min))
    flow_15_min = pd.concat(yearly_flows).sort_index()

    full_adjacency = np.load(raw_dir / "ca_rn_adj.npy")
    adjacency_sd = full_adjacency[np.ix_(sd_row_positions, sd_row_positions)]
    assert adjacency_sd.shape == (EXPECTED_NUM_SENSORS, EXPECTED_NUM_SENSORS)

    np.savez_compressed(
        processed_dir / "sd_flow_15min.npz",
        flow=flow_15_min.to_numpy(dtype=np.float32),
        timestamps=flow_15_min.index.to_numpy().astype("datetime64[m]"),
        sensor_ids=meta_sd["ID"].to_numpy(),
        adjacency=adjacency_sd.astype(np.float32),
    )
    meta_sd.to_csv(processed_dir / "sd_meta.csv", index=False)


def load_processed(processed_dir):
    """Return flow (T, N) with NaNs, timestamps (DatetimeIndex), adjacency, meta."""
    processed_dir = Path(processed_dir)
    cached = np.load(processed_dir / "sd_flow_15min.npz")
    timestamps = pd.DatetimeIndex(cached["timestamps"])
    meta_sd = pd.read_csv(processed_dir / "sd_meta.csv")
    return cached["flow"], timestamps, cached["adjacency"], meta_sd


def missing_value_report(flow):
    """Per-sensor missing fraction and overall missing fraction (Q3a item 4)."""
    missing_per_sensor = np.isnan(flow).mean(axis=0)
    return {
        "overall_missing_fraction": float(np.isnan(flow).mean()),
        "sensors_with_any_missing": int((missing_per_sensor > 0).sum()),
        "worst_sensor_missing_fraction": float(missing_per_sensor.max()),
        "zero_fraction": float(np.nanmean(flow == 0)),
    }


def forward_fill_short_gaps(flow, max_gap=MAX_FILL_GAP):
    """Forward-fill NaN runs of up to max_gap steps in the INPUT array.

    Uses only past values, so it leaks nothing across splits. The original NaN
    mask is kept separately and is always used to exclude entries from loss/metrics.
    """
    return pd.DataFrame(flow).ffill(limit=max_gap).to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------
# Q4(a): splits, normalisation, windows
# --------------------------------------------------------------------------
class SplitArrays:
    """Holds flow and validity mask for one chronological split."""

    def __init__(self, flow, timestamps):
        self.raw_flow = flow                          # (T, N), NaN = missing
        self.valid_mask = ~np.isnan(flow)             # True where observed
        self.timestamps = timestamps


def split_chronologically(flow, timestamps):
    slices = {
        "train": timestamps <= pd.Timestamp(TRAIN_END),
        "val": (timestamps >= pd.Timestamp(VAL_START)) & (timestamps <= pd.Timestamp(VAL_END)),
        "test": (timestamps >= pd.Timestamp(TEST_START)) & (timestamps <= pd.Timestamp(TEST_END)),
    }
    return {name: SplitArrays(flow[sel], timestamps[sel]) for name, sel in slices.items()}


def fit_normaliser(train_flow):
    """Mean/std from observed TRAIN entries only."""
    mean = float(np.nanmean(train_flow))
    std = float(np.nanstd(train_flow))
    return mean, std


class WindowDataset(Dataset):
    """Sliding windows within ONE split (so no window crosses a split boundary).

    Item: input (INPUT_LENGTH, N, 1), target (HORIZON, N), target_mask (HORIZON, N).
    Input is forward-filled (short gaps), normalised, remaining NaN -> 0 (= train mean).
    Target is normalised; target_mask marks genuinely observed entries.
    """

    def __init__(self, split, mean, std, stride=1):
        filled = forward_fill_short_gaps(split.raw_flow)
        normalised_input = (filled - mean) / std
        self.input_series = np.nan_to_num(normalised_input, nan=0.0).astype(np.float32)
        self.target_series = np.nan_to_num(
            (split.raw_flow - mean) / std, nan=0.0
        ).astype(np.float32)
        self.valid_mask = split.valid_mask
        window_length = INPUT_LENGTH + HORIZON
        num_windows = split.raw_flow.shape[0] - window_length + 1
        self.window_starts = np.arange(0, max(num_windows, 0), stride)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        start = self.window_starts[index]
        input_end = start + INPUT_LENGTH
        target_end = input_end + HORIZON
        history = self.input_series[start:input_end]                # (8, N)
        target = self.target_series[input_end:target_end]           # (4, N)
        target_mask = self.valid_mask[input_end:target_end]         # (4, N)
        return (
            torch.from_numpy(history).unsqueeze(-1),                # (8, N, 1)
            torch.from_numpy(target),
            torch.from_numpy(target_mask.astype(np.float32)),
        )


# --------------------------------------------------------------------------
# Q3(b)/(c): graph helpers
# --------------------------------------------------------------------------
def graph_summary(adjacency):
    """Nodes, nonzero edges, symmetry, degree stats (Q3b items 1, 2, 4)."""
    nonzero = adjacency != 0
    degrees = nonzero.sum(axis=1)
    return {
        "num_nodes": int(adjacency.shape[0]),
        "num_nonzero_entries": int(nonzero.sum()),
        "num_self_loops": int(np.trace(nonzero)),
        "is_symmetric": bool(np.allclose(adjacency, adjacency.T)),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
    }


def k_hop_neighbours(adjacency, node_index, num_hops):
    """Nodes reachable from node_index within num_hops edges (excluding itself)."""
    reachable = {node_index}
    frontier = {node_index}
    for _ in range(num_hops):
        next_frontier = set()
        for node in frontier:
            next_frontier |= set(np.flatnonzero(adjacency[node] != 0).tolist())
        frontier = next_frontier - reachable
        reachable |= next_frontier
    reachable.discard(node_index)
    return sorted(reachable)


def normalise_adjacency(adjacency, mode):
    """Add self-loops and normalise.

    mode='sym': D^-1/2 (A+I) D^-1/2   (valid for symmetric, nonnegative A)
    mode='row': D^-1 (A+I)            (row-stochastic; also fine for directed A)
    """
    with_self_loops = adjacency + np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    degree = with_self_loops.sum(axis=1)
    if mode == "sym":
        inv_sqrt_degree = 1.0 / np.sqrt(degree)
        return inv_sqrt_degree[:, None] * with_self_loops * inv_sqrt_degree[None, :]
    if mode == "row":
        return with_self_loops / degree[:, None]
    raise ValueError(f"Unknown normalisation mode: {mode}")
