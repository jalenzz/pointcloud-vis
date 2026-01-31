"""Utility functions for point cloud visualization."""

import numpy as np
import open3d as o3d

from .constants import CLASS_NAMES, CLASS_COLORS


def create_point_cloud(xyz: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def print_class_legend() -> None:
    """Print semantic class legend with colors."""
    print("\n" + "=" * 50)
    print("Semantic Class Legend:")
    print("=" * 50)
    for i, name in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS[i]
        print(f"  {i:2d}: {name:12s} RGB({r:.1f}, {g:.1f}, {b:.1f})")
    print("=" * 50 + "\n")


def camera_params_equal(params1, params2, tolerance: float = 1e-6) -> bool:
    """Check if two camera parameters are equal within tolerance."""
    if params1 is None or params2 is None:
        return False
    ext1 = np.asarray(params1.extrinsic)
    ext2 = np.asarray(params2.extrinsic)
    return np.allclose(ext1, ext2, atol=tolerance)
