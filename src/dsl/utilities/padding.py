import numpy as np

# This file contains utility functions to pad and remove padding from grids.

# Implemented utility methods:
# - pad_grid(grid, desired_dim): Pad the grid to the desired dimensions.
# - remove_padding(grid): Extract the original grid from the padded grid.

def pad_grid(grid: np.ndarray, desired_dim: tuple[int, int]=(30, 30)) -> np.ndarray:
    """
    Pad the grid to the desired dimensions.
    """
    desired_rows, desired_cols = desired_dim
    n_rows, n_cols = grid.shape
    padded_grid = - np.ones((desired_rows, desired_cols), dtype=int)
    padded_grid[:n_rows, :n_cols] = grid
    return padded_grid

def unpad_grid(grid: np.ndarray) -> np.ndarray:
    """
    Extract the original grid from the padded grid while preserving its original shape.
    """
    mask = (grid != -1)
    rows, cols = np.where(mask)
    if rows.size == 0 and cols.size == 0:
        return np.zeros((0, 0), dtype=int)
    unpadded = np.zeros((max(rows)+1, max(cols)+1), dtype=int)
    unpadded[rows, cols] = grid[mask]
    return unpadded