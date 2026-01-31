# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Merge blocks back into complete rooms (auto-processes all data directories)
uv run python merge_blocks.py

# Run the visualizer (interactive menu to select merged data)
uv run python visualize.py
```

Always use `uv` to run Python commands.

## Data Format

Data is stored in `./data/{experiment}/` subdirectories (e.g., 10-3, 12-1, 8-5). Expected npy files (S3DIS dataset format):
- `{block_name}_points.npy` - shape: (N, 6), columns: xyz (0:3), rgb (3:6, values 0-255)
- `{block_name}_gt.npy` - shape: (N,), ground truth labels (0-12)
- `{block_name}_pred.npy` - shape: (N,), predicted labels (0-12)

## Architecture

```
viewer/              # Core visualization package
├── __init__.py      # Exports PointCloudViewer, CLASS_NAMES, CLASS_COLORS
├── constants.py     # CLASS_NAMES, CLASS_COLORS
├── data.py          # Data loading functions
├── utils.py         # Helper functions
└── viewer.py        # PointCloudViewer class

visualize.py         # Entry point (interactive menu)
merge_blocks.py      # Merge tool (auto-processes ./data subdirectories)
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| N/P | Next/Previous room |
| 1-5 | Switch visualization mode (single-window only) |
| [/] | Decrease/Increase hide-top percentage (default 50%) |
| +/- | Adjust point size |
| R | Reset view |
| S | Save transparent screenshots |
| Q | Quit |

## Visualization Modes

1. RGB - Original colors
2. Ground Truth - Semantic labels
3. Prediction - Predicted labels
4. RGB \| GT \| Pred - Three-window comparison
5. GT \| Pred - Two-window comparison (default)
