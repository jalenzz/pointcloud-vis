"""S3DIS dataset semantic class definitions and color mappings."""

import numpy as np

CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]

CLASS_COLORS = np.array([
    [0.78, 0.86, 0.94],  # ceiling - light blue
    [0.36, 0.47, 0.67],  # floor - blue gray
    [0.80, 0.80, 0.80],  # wall - neutral gray
    [0.55, 0.55, 0.58],  # beam - steel gray
    [0.38, 0.38, 0.40],  # column - dark gray
    [0.47, 0.64, 0.80],  # window - cool blue
    [0.69, 0.58, 0.45],  # door - warm brown
    [0.90, 0.71, 0.37],  # table - warm orange
    [0.86, 0.42, 0.47],  # chair - soft red
    [0.66, 0.46, 0.70],  # sofa - purple gray
    [0.46, 0.72, 0.64],  # bookcase - teal
    [0.42, 0.64, 0.46],  # board - green
    [0.73, 0.73, 0.73],  # clutter - light gray
])
