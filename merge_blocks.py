"""Merge block data back into complete rooms."""

import glob
import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def get_data_directories(base_path: str = './data') -> list:
    """Find all data subdirectories containing block files."""
    if not os.path.isdir(base_path):
        return []

    dirs = []
    for name in os.listdir(base_path):
        subdir = os.path.join(base_path, name)
        if os.path.isdir(subdir) and not name.endswith('_merged'):
            if glob.glob(os.path.join(subdir, '*_points.npy')):
                dirs.append(subdir)
    return sorted(dirs)


def get_room_blocks(data_path: str) -> dict:
    """Group block files by room name.

    Returns:
        dict: {room_name: [(block_id, points_file, gt_file, pred_file), ...]}
    """
    files = glob.glob(os.path.join(data_path, '*_points.npy'))
    rooms = defaultdict(list)

    for points_file in files:
        basename = os.path.basename(points_file)
        match = re.match(r'(.+)_block_(\d+)_points\.npy', basename)
        if not match:
            continue

        room_name = match.group(1)
        block_id = int(match.group(2))
        gt_file = points_file.replace('_points.npy', '_gt.npy')
        pred_file = points_file.replace('_points.npy', '_pred.npy')

        if os.path.exists(gt_file) and os.path.exists(pred_file):
            rooms[room_name].append((block_id, points_file, gt_file, pred_file))

    for room_name in rooms:
        rooms[room_name].sort(key=lambda x: x[0])

    return rooms


def merge_room_blocks(
    block_list: list,
    deduplicate: bool = False,
    tolerance: float = 1e-6
) -> tuple:
    """Merge all blocks of a room into one.

    Args:
        block_list: List of (block_id, points_file, gt_file, pred_file)
        deduplicate: If True, remove duplicate points based on coordinates
        tolerance: Distance threshold for considering points as duplicates

    Returns:
        Tuple of (merged_points, merged_gt, merged_pred)
    """
    all_points = []
    all_gt = []
    all_pred = []

    for _, points_file, gt_file, pred_file in block_list:
        all_points.append(np.load(points_file))
        all_gt.append(np.load(gt_file))
        all_pred.append(np.load(pred_file))

    merged_points = np.concatenate(all_points, axis=0)
    merged_gt = np.concatenate(all_gt)
    merged_pred = np.concatenate(all_pred)

    if deduplicate:
        xyz = merged_points[:, :3]
        scale = int(1 / tolerance)
        xyz_scaled = (xyz * scale).astype(np.int64)
        _, unique_indices = np.unique(xyz_scaled, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)

        original_count = len(xyz)
        merged_points = merged_points[unique_indices]
        merged_gt = merged_gt[unique_indices]
        merged_pred = merged_pred[unique_indices]
        print(f'    Deduplication: {original_count} -> {len(unique_indices)} points')

    return merged_points, merged_gt, merged_pred


def process_directory(
    data_path: str,
    deduplicate: bool = False,
    tolerance: float = 1e-4
) -> str:
    """Process a single data directory."""
    output_path = data_path.rstrip('/\\') + '_merged'
    os.makedirs(output_path, exist_ok=True)

    rooms = get_room_blocks(data_path)
    print(f'Found {len(rooms)} rooms to merge')

    for room_name, block_list in tqdm(rooms.items(), desc='Merging rooms'):
        print(f'\n{room_name}: {len(block_list)} blocks')

        merged_points, merged_gt, merged_pred = merge_room_blocks(
            block_list, deduplicate=deduplicate, tolerance=tolerance
        )

        print(f'  Total points: {merged_points.shape[0]}')

        np.save(os.path.join(output_path, f'{room_name}_points.npy'), merged_points)
        np.save(os.path.join(output_path, f'{room_name}_gt.npy'), merged_gt)
        np.save(os.path.join(output_path, f'{room_name}_pred.npy'), merged_pred)

    print(f'Merged data saved to: {output_path}')
    return output_path


def main() -> None:
    data_dirs = get_data_directories('./data')

    if not data_dirs:
        print('No data directories found in ./data/')
        return

    print(f'Found {len(data_dirs)} data directories:')
    for d in data_dirs:
        print(f'  - {d}')
    print()

    for data_path in data_dirs:
        print('=' * 60)
        print(f'Processing: {data_path}')
        print('=' * 60)
        process_directory(data_path)
        print()

    print('All directories processed.')


if __name__ == '__main__':
    main()
