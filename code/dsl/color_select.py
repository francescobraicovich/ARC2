import numpy as np
from dsl.utilities.checks import check_color_rank, check_integer
from scipy.ndimage import find_objects, label

# ColorSelector class that contains methods to select colors from a grid.

# Implemented methods:
# - mostcolor(grid): Return the most common color in the grid.
# - leastcolor(grid): Return the least common color in the grid.
# - rankcolor(grid, rank): Return the rank-th common color in the grid.
# - rank_largest_shape_color_nodiag(grid, rank): Return the color of the rank-th largest shape in the grid without considering diagonal connections.
# - rank_largest_shape_color_diag(grid, rank): Return the color of the rank-th largest shape in the grid considering diagonal connections.
# - color_number(grid, color): Return the color number if the color is not in the grid. This method should only be used when the color is not in the grid.

class ColorSelector:
    def __init__(self, num_colors: int = 9):
        self.num_colors = num_colors
        self.all_colors = np.arange(num_colors)
        self.invalid_color = -10 # Return this value if the color is invalid so the selection doesn't select anything.
        self.big_number = 1000000 # A big number to use for sorting colors by decreasing count (otherwise colors wihth 0 count will be selected).

    def mostcolor(self, grid: np.ndarray) -> int:
        """ most common color """
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        return np.argmax(counts)
    
    def leastcolor(self, grid: np.ndarray) -> int:
        """ least common color """
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        counts[counts == 0] = self.big_number
        return np.argmin(counts)
    
    def rankcolor(self, grid: np.ndarray, rank: int) -> int:
        """ rank-th common color """
        if check_color_rank(rank) == False:
            return self.invalid_color
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        return np.argsort(-counts)[rank]
    
    def rank_largest_shape_color_nodiag(self, grid: np.ndarray, rank: int) -> int:
        """ the color of the rank-th largest shape """
        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)
        if check_integer(rank, 0, num_colors) == False:
            return self.invalid_color
        dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)
        for i in range(num_colors):
            copied_grid = np.copy(grid)
            color_mask = copied_grid == unique_colors[i]
            copied_grid[color_mask] = 1
            copied_grid[~color_mask] = 0
            labeled_grid, num_labels = label(copied_grid)
            unique, counts = np.unique(labeled_grid, return_counts=True)
            unique, counts = unique[1:], counts[1:] # Remove the 0 label (background)
            biggest_count = np.max(counts)
            dimension_of_biggest_shape[i] = biggest_count

        sorted_indices = np.argsort(-dimension_of_biggest_shape)
        index = sorted_indices[rank]
        color = unique_colors[index]
        return color

    def rank_largest_shape_color_diag(self, grid: np.ndarray, rank: int) -> int:
        """ The color of the rank-th largest shape, considering diagonal connections """
        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)
        if not check_integer(rank, 0, num_colors):
            return self.invalid_color
        
        dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)
        
        # Define connectivity for diagonal connections (8-connectivity for 2D grids)
        connectivity = np.ones((3, 3), dtype=int)  # A 3x3 grid of ones includes diagonals
        
        for i, color in enumerate(unique_colors):
            copied_grid = np.copy(grid)
            color_mask = copied_grid == color
            copied_grid[color_mask] = 1
            copied_grid[~color_mask] = 0
            
            # Label connected regions with diagonal connectivity
            labeled_grid, num_labels = label(copied_grid, structure=connectivity)
            unique, counts = np.unique(labeled_grid, return_counts=True)
            unique, counts = unique[1:], counts[1:]  # Remove the 0 label (background)
            biggest_count = np.max(counts)
            dimension_of_biggest_shape[i] = biggest_count
        
        sorted_indices = np.argsort(-dimension_of_biggest_shape)
        index = sorted_indices[rank]
        color = unique_colors[index]
        return color
    
    def color_number(self, grid: np.ndarray, color: int) -> int:
        """ Select the number of cells with the given color only if the color is not in the grid """
        if check_integer(color, 0, self.num_colors) == False:
            return self.invalid_color
        unique_colors = np.unique(grid)
        if color in unique_colors:
            return self.invalid_color
        return color

    def palette(self, grid: np.ndarray) -> set:
        """Return the set of unique colors in the grid."""
        return set(np.unique(grid))
    
    def numcolors(self, grid: np.ndarray) -> int:
        """Return the number of unique colors in the grid."""
        return len(palette(grid))
    
    def colorcount(self, grid: np.ndarray, value: int) -> int:
        """Return the number of cells with the given color."""
        return np.sum(grid == value)
    

    