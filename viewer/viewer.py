"""Point Cloud Viewer with keyboard navigation and multi-window synchronization."""

import os

import numpy as np
import open3d as o3d
from PIL import Image

from .data import load_point_cloud_data, get_label_colors, calculate_accuracy
from .utils import create_point_cloud, camera_params_equal


class PointCloudViewer:
    """Point cloud viewer with keyboard navigation. Modes 4/5 use multi-window sync."""

    MULTI_WINDOW_MODES = {"4", "5"}
    MODE_NAMES = {
        "1": "RGB",
        "2": "Ground Truth",
        "3": "Prediction",
        "4": "RGB | GT | Pred",
        "5": "GT | Pred",
    }

    def __init__(
        self,
        data_dir: str,
        blocks: list,
        start_idx: int = 0,
        mode: str = "5",
        hide_top_percent: float = 50,
    ):
        self.data_dir = data_dir
        self.blocks = blocks
        self.current_idx = start_idx
        self.mode = mode
        self.hide_top_percent = hide_top_percent
        self.point_size = 5.0
        self.running = True
        self.current_xyz = None
        self.visualizers = []
        self.geometries = []
        self.window_names = []
        self.last_cam_params = []

    def is_multi_window_mode(self) -> bool:
        return self.mode in self.MULTI_WINDOW_MODES

    def get_mode_name(self) -> str:
        return self.MODE_NAMES.get(self.mode, "Unknown")

    def _get_window_configs(self, rgb, gt, pred) -> list:
        """Return list of (window_name, colors) for the current mode."""
        gt_colors = get_label_colors(gt)
        pred_colors = get_label_colors(pred)
        configs = {
            "1": [("Main", rgb)],
            "2": [("Main", gt_colors)],
            "3": [("Main", pred_colors)],
            "4": [("rgb", rgb), ("gt", gt_colors), ("pred", pred_colors)],
            "5": [("gt", gt_colors), ("pred", pred_colors)],
        }
        return configs.get(self.mode, configs["5"])

    def load_current(self) -> tuple:
        """Load point cloud data for the current index."""
        block_name = self.blocks[self.current_idx]
        xyz, rgb, gt, pred = load_point_cloud_data(self.data_dir, block_name)

        if self.hide_top_percent > 0:
            z_range = xyz[:, 2].max() - xyz[:, 2].min()
            z_threshold = xyz[:, 2].max() - z_range * (self.hide_top_percent / 100.0)
            mask = xyz[:, 2] < z_threshold
            xyz, rgb, gt, pred = xyz[mask], rgb[mask], gt[mask], pred[mask]

        accuracy = calculate_accuracy(gt, pred)
        print(f"\n[{self.current_idx + 1}/{len(self.blocks)}] {block_name}")
        print(f"  Points: {len(xyz)} | Accuracy: {accuracy:.2f}% | "
              f"Mode: {self.get_mode_name()} | Hide top: {self.hide_top_percent:.0f}%")

        return xyz, rgb, gt, pred, block_name, accuracy

    def create_windows(self, xyz, rgb, gt, pred) -> None:
        """Create visualization windows for the current mode."""
        self.current_xyz = xyz
        self._close_all_windows()

        configs = self._get_window_configs(rgb, gt, pred)
        num_windows = len(configs)
        is_multi = self.is_multi_window_mode()
        win_width = 2560 // num_windows if is_multi else 1920
        win_height = 1000 if is_multi else 1080

        for i, (name, colors) in enumerate(configs):
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(
                window_name=name if is_multi else "Point Cloud Viewer",
                width=win_width,
                height=win_height,
                left=i * win_width if is_multi else 50,
                top=50,
            )

            pcd = create_point_cloud(xyz, colors)
            vis.add_geometry(pcd)
            self._register_callbacks(vis)

            self.visualizers.append(vis)
            self.geometries.append([pcd])
            self.window_names.append(name)
            self.last_cam_params.append(None)

        self._set_default_view()
        self._apply_point_size()

    def _close_all_windows(self) -> None:
        """Close all existing visualizer windows and reset state."""
        for vis in self.visualizers:
            vis.destroy_window()
        self.visualizers = []
        self.geometries = []
        self.window_names = []
        self.last_cam_params = []

    def _set_default_view(self) -> None:
        """Set default camera view: observe from upper diagonal toward point cloud center."""
        if self.current_xyz is None or len(self.current_xyz) == 0:
            return

        center = (self.current_xyz.min(axis=0) + self.current_xyz.max(axis=0)) / 2
        for vis in self.visualizers:
            ctrl = vis.get_view_control()
            ctrl.set_lookat(center)
            ctrl.set_front([1, 1, 1])
            ctrl.set_up([0, 0, 1])
            ctrl.set_zoom(0.5)

    def _apply_point_size(self) -> None:
        """Apply current point size to all windows."""
        for vis in self.visualizers:
            vis.get_render_option().point_size = self.point_size

    def _register_callbacks(self, vis) -> None:
        """Register keyboard callbacks on a visualizer window."""
        nav_keys = [
            ('N', self.next_callback),
            ('P', self.prev_callback),
            ('Q', self.quit_callback),
            ('R', self.reset_view_callback),
            ('S', self.save_screenshots_callback),
        ]
        for key, callback in nav_keys:
            vis.register_key_callback(ord(key.upper()), callback)
            vis.register_key_callback(ord(key.lower()), callback)

        vis.register_key_callback(ord('+'), self.increase_point_size_callback)
        vis.register_key_callback(ord('='), self.increase_point_size_callback)
        vis.register_key_callback(ord('-'), self.decrease_point_size_callback)
        vis.register_key_callback(ord('['), self.decrease_hide_top_callback)
        vis.register_key_callback(ord(']'), self.increase_hide_top_callback)

        if not self.is_multi_window_mode():
            for mode_key in ["1", "2", "3", "4", "5"]:
                vis.register_key_callback(ord(mode_key), self._make_mode_callback(mode_key))

    def update_geometries(self) -> None:
        """Replace point cloud geometries in all windows with freshly loaded data."""
        xyz, rgb, gt, pred, _, _ = self.load_current()
        self.current_xyz = xyz

        configs = self._get_window_configs(rgb, gt, pred)
        for i, (vis, (_, colors)) in enumerate(zip(self.visualizers, configs)):
            for geom in self.geometries[i]:
                vis.remove_geometry(geom, reset_bounding_box=False)
            pcd = create_point_cloud(xyz, colors)
            vis.add_geometry(pcd, reset_bounding_box=True)
            self.geometries[i] = [pcd]

        self._set_default_view()

    def sync_cameras(self) -> None:
        """Sync camera across all windows - detect which window changed and propagate."""
        if len(self.visualizers) < 2:
            return

        current_params = [
            vis.get_view_control().convert_to_pinhole_camera_parameters()
            for vis in self.visualizers
        ]

        changed_idx = next(
            (i for i, (cur, last) in enumerate(zip(current_params, self.last_cam_params))
             if not camera_params_equal(cur, last)),
            -1
        )

        if changed_idx >= 0:
            master_params = current_params[changed_idx]
            for i, vis in enumerate(self.visualizers):
                if i != changed_idx:
                    vis.get_view_control().convert_from_pinhole_camera_parameters(
                        master_params, allow_arbitrary=True
                    )
            self.last_cam_params = [master_params] * len(self.visualizers)
        else:
            self.last_cam_params = current_params

    def next_callback(self, vis) -> bool:
        self.current_idx = (self.current_idx + 1) % len(self.blocks)
        self.update_geometries()
        return False

    def prev_callback(self, vis) -> bool:
        self.current_idx = (self.current_idx - 1) % len(self.blocks)
        self.update_geometries()
        return False

    def quit_callback(self, vis) -> bool:
        print("\nExiting visualization")
        self.running = False
        return False

    def reset_view_callback(self, vis) -> bool:
        for v in self.visualizers:
            v.reset_view_point(True)
        return False

    def increase_hide_top_callback(self, vis) -> bool:
        self.hide_top_percent = min(90, self.hide_top_percent + 10)
        print(f"  Hide top: {self.hide_top_percent:.0f}%")
        self.update_geometries()
        return False

    def decrease_hide_top_callback(self, vis) -> bool:
        self.hide_top_percent = max(0, self.hide_top_percent - 10)
        print(f"  Hide top: {self.hide_top_percent:.0f}%")
        self.update_geometries()
        return False

    def increase_point_size_callback(self, vis) -> bool:
        self.point_size = min(50.0, self.point_size + 1.0)
        self._apply_point_size()
        print(f"  Point size: {self.point_size:.1f}")
        return False

    def decrease_point_size_callback(self, vis) -> bool:
        self.point_size = max(0.5, self.point_size - 1.0)
        self._apply_point_size()
        print(f"  Point size: {self.point_size:.1f}")
        return False

    def save_screenshots_callback(self, vis) -> bool:
        """Save screenshots with transparent background using chroma key."""
        block_name = self.blocks[self.current_idx]
        save_dir = os.path.join(self.data_dir, "screenshots")
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        for vis_i, win_name in zip(self.visualizers, self.window_names):
            bg_color = np.array(vis_i.get_render_option().background_color)

            vis_i.poll_events()
            vis_i.update_renderer()
            img_arr = np.asarray(vis_i.capture_screen_float_buffer(do_render=True))

            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)
            img_uint8 = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)

            chroma_uint8 = (bg_color * 255).astype(np.uint8)
            is_bg = np.all(np.abs(img_uint8.astype(np.int16) - chroma_uint8) <= 15, axis=-1)
            alpha = np.where(is_bg, 0, 255).astype(np.uint8)
            rgba = np.dstack([img_uint8, alpha])

            safe_win = win_name.replace(" ", "_")
            filepath = os.path.join(save_dir, f"{block_name}_{safe_win}.png")
            Image.fromarray(rgba, mode="RGBA").save(filepath)
            saved_files.append(filepath)

        print(f"  Saved {len(saved_files)} screenshots to: {save_dir}")
        for f in saved_files:
            print(f"    - {os.path.basename(f)}")
        return False

    def _make_mode_callback(self, mode: str):
        """Create a callback that switches to the given visualization mode."""
        def callback(vis) -> bool:
            self.mode = mode
            print(f"  Switched to mode: {self.get_mode_name()}")
            if mode in self.MULTI_WINDOW_MODES:
                xyz, rgb, gt, pred, _, _ = self.load_current()
                self.create_windows(xyz, rgb, gt, pred)
            else:
                self.update_geometries()
            return False
        return callback

    def run(self) -> None:
        """Run the visualization loop."""
        xyz, rgb, gt, pred, _, _ = self.load_current()
        self.create_windows(xyz, rgb, gt, pred)

        print("\n" + "=" * 50)
        print("Controls:")
        print("  N: Next point cloud")
        print("  P: Previous point cloud")
        print("  R: Reset view")
        print("  [/]: Adjust hide-top percentage (remove ceiling)")
        print("  +/-: Adjust point size")
        print("  S: Save transparent screenshots")
        print("  Q: Quit")
        if self.is_multi_window_mode():
            print("  (All windows sync automatically)")
        else:
            print("  1-3: Switch to single-view mode")
            print("  4-5: Switch to multi-window comparison mode")
        print("=" * 50)

        while self.running:
            if not all(vis.poll_events() for vis in self.visualizers):
                break
            if self.is_multi_window_mode():
                self.sync_cameras()
            for vis in self.visualizers:
                vis.update_renderer()

        for vis in self.visualizers:
            vis.destroy_window()
