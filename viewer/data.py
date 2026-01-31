"""Data loading utilities for point cloud files."""

import glob
import os

import numpy as np

from .constants import CLASS_COLORS


def load_point_cloud_data(data_dir: str, block_name: str) -> tuple:
    """Load point cloud data and labels for a block."""
    base = os.path.join(data_dir, block_name)
    points = np.load(f"{base}_points.npy")
    gt = np.load(f"{base}_gt.npy")
    pred = np.load(f"{base}_pred.npy")

    xyz = points[:, :3]
    rgb = points[:, 3:6]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    return xyz, rgb, gt, pred


def get_available_blocks(data_dir: str) -> list:
    """Return sorted list of block names found in data_dir."""
    files = glob.glob(os.path.join(data_dir, "*_points.npy"))
    return sorted(os.path.basename(f).replace("_points.npy", "") for f in files)


def get_label_colors(labels: np.ndarray) -> np.ndarray:
    """Map semantic labels to RGB colors via CLASS_COLORS lookup."""
    safe_labels = np.clip(labels, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[safe_labels]


def calculate_accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculate prediction accuracy as a percentage."""
    return np.mean(gt == pred) * 100
