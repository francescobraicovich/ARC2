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
    def __init__(self, num_colors: int = 10):
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
        """Rank-th common color"""
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("Rank must be a non-negative integer.")
        
        # Get unique colors and their counts
        unique_colors, counts = np.unique(grid, return_counts=True)
        sorted_indices = np.argsort(-counts)  # Indices of colors sorted by descending frequency
        
        if rank < len(unique_colors):
            # Return the color corresponding to the rank-th most common occurrence
            return unique_colors[sorted_indices[rank]]
        else:
            # If rank exceeds the number of unique colors, return the last color with a count > 0
            last_nonzero_index = sorted_indices[-1]  # Last index in sorted order
            return unique_colors[last_nonzero_index]
    
    def rank_largest_shape_color_nodiag(self, grid: np.ndarray, rank: int) -> int:
        """ the color of the rank-th largest shape """
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("Rank must be a non-negative integer.")
        if not np.issubdtype(grid.dtype, np.integer) or grid.size == 0:
            raise ValueError("Grid must be a non-empty array of integers.")
        
        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)
        dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)
        
        for i, color in enumerate(unique_colors):
            color_mask = grid == color
            labeled_grid, num_labels = label(color_mask.astype(int))
            _, counts = np.unique(labeled_grid, return_counts=True)
            counts = counts[1:]  # Exclude background (label 0)
            dimension_of_biggest_shape[i] = np.max(counts) if len(counts) > 0 else 0
        
        sorted_indices = np.argsort(-dimension_of_biggest_shape)
        if rank >= len(sorted_indices):
            smallest_non_zero_index = np.argmin(
                np.where(dimension_of_biggest_shape > 0, dimension_of_biggest_shape, np.inf)
            )
        
            return unique_colors[smallest_non_zero_index] if dimension_of_biggest_shape[smallest_non_zero_index] > 0 else unique_colors[0]
        
        return unique_colors[sorted_indices[rank]]

    def rank_largest_shape_color_diag(self, grid: np.ndarray, rank: int) -> int:
        """ The color of the rank-th largest shape, considering diagonal connections """
        raise DeprecationWarning("This method is deprecated. Use rank_largest_shape_color_nodiag instead.")
        
        # Find the unique colors in the grid
        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)
        
        # Ensure rank is valid
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("Rank must be a non-negative integer.")
        
        dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)
        
        # Define connectivity for diagonal connections (8-connectivity for 2D grids)
        connectivity = np.ones((3, 3), dtype=int)  # A 3x3 grid of ones includes diagonals
        
        # Loop over all unique colors and label the connected components
        for i, color in enumerate(unique_colors):
            copied_grid = np.copy(grid)
            # Create a mask where the current color exists in the grid
            color_mask = copied_grid == color
            copied_grid[color_mask] = 1
            copied_grid[~color_mask] = 0
            # Label connected regions with diagonal connectivity
            labeled_grid, num_labels = label(copied_grid, structure=connectivity)
            unique, counts = np.unique(labeled_grid, return_counts=True)
            # Remove the background label (0) from the counts
            unique, counts = unique[1:], counts[1:]
            # Handle empty components
            biggest_count = np.max(counts) if len(counts) > 0 else 0
            # Store the size of the largest connected component for that color
            dimension_of_biggest_shape[i] = biggest_count
        
        # Sort the colors based on the size of their largest connected component (descending order)
        sorted_indices = np.argsort(-dimension_of_biggest_shape)
        
        # Handle out-of-bound ranks by falling back to the smallest non-zero shape
        if rank >= len(sorted_indices):
            non_zero_sizes = dimension_of_biggest_shape[dimension_of_biggest_shape > 0]
            if len(non_zero_sizes) == 0:  # If no non-zero shapes exist
                return 0
            smallest_non_zero_index = np.argmin(dimension_of_biggest_shape + (dimension_of_biggest_shape == 0) * np.max(dimension_of_biggest_shape))
            return unique_colors[smallest_non_zero_index]

        # Otherwise, return the color corresponding to the rank-th largest shape
        index = sorted_indices[rank]
        return unique_colors[index]
    
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
    

    