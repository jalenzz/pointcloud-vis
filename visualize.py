"""Point Cloud Semantic Segmentation Visualization using Open3D."""

import glob
import os

from viewer import PointCloudViewer
from viewer.data import get_available_blocks
from viewer.utils import print_class_legend


def get_merged_directories(base_path: str = './data') -> list:
    """Find all merged data directories."""
    if not os.path.isdir(base_path):
        return []

    dirs = []
    for name in os.listdir(base_path):
        subdir = os.path.join(base_path, name)
        if os.path.isdir(subdir) and name.endswith('_merged'):
            if glob.glob(os.path.join(subdir, '*_points.npy')):
                dirs.append(subdir)
    return sorted(dirs)


def prompt_selection(prompt: str, max_idx: int, default: int = 0) -> int:
    """Prompt user to select an index, returning default on empty/invalid input."""
    user_input = input(prompt).strip()
    if user_input.lower() == 'q':
        return -1
    try:
        idx = int(user_input) if user_input else default
        return idx if 0 <= idx < max_idx else default
    except ValueError:
        return default


def main() -> None:
    merged_dirs = get_merged_directories('./data')

    if not merged_dirs:
        print('No merged data directories found.')
        print('Please run: uv run python merge_blocks.py')
        return

    print('Available data directories:')
    for i, d in enumerate(merged_dirs):
        blocks = get_available_blocks(d)
        print(f'  {i}: {os.path.basename(d)} ({len(blocks)} rooms)')

    print('\nEnter number to select directory (default: 0), or q to quit:')
    dir_idx = prompt_selection('> ', len(merged_dirs))
    if dir_idx < 0:
        return

    data_dir = merged_dirs[dir_idx]
    blocks = get_available_blocks(data_dir)

    print(f'\nSelected: {os.path.basename(data_dir)}')
    print(f'Found {len(blocks)} rooms')

    print_class_legend()

    print('Available rooms (first 10):')
    for i, block in enumerate(blocks[:10]):
        print(f'  {i}: {block}')
    if len(blocks) > 10:
        print(f'  ... and {len(blocks) - 10} more')

    print('\nEnter number to select starting room (default: 0):')
    start_idx = prompt_selection('> ', len(blocks))
    if start_idx < 0:
        start_idx = 0

    print('\nVisualization modes:')
    print('  1: RGB original colors')
    print('  2: Ground Truth semantic labels')
    print('  3: Prediction labels')
    print('  4: Multi-window comparison (RGB | GT | Pred)')
    print('  5: Multi-window comparison (GT | Pred)')
    mode = input('Select initial mode (default: 5): ').strip() or '5'
    if mode not in ['1', '2', '3', '4', '5']:
        mode = '5'

    viewer = PointCloudViewer(data_dir, blocks, start_idx, mode)
    viewer.run()


if __name__ == '__main__':
    main()
