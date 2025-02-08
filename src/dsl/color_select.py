import numpy as np
from dsl.utilities.checks import check_color_rank, check_integer
from scipy.ndimage import find_objects, label


class ColorSelector:
    """
    Class for selecting colors from a grid using various strategies.

    Implemented methods:
      - mostcolor: Returns the most common color in the grid.
      - leastcolor: Returns the least common color in the grid.
      - rankcolor: Returns the rank-th most common color in the grid.
      - rank_largest_shape_color_nodiag: Returns the color of the rank-th largest connected shape
                                           (without considering diagonal connectivity).
      - rank_largest_shape_color_diag: Returns the color of the rank-th largest connected shape
                                         (with diagonal connectivity; deprecated).
      - color_number: Returns a given color number if that color is not present in the grid.
    """

    def __init__(self, num_colors: int = 10):
        """
        Initialize the ColorSelector.

        Args:
            num_colors (int): Total number of possible colors. Default is 10.
        """
        self.num_colors = num_colors
        self.all_colors = np.arange(num_colors)
        self.invalid_color = -10  # Returned when the color is invalid or present in the grid.
        self.big_number = 1000000  # Large number used to penalize zero counts when sorting.

    def mostcolor(self, grid: np.ndarray) -> int:
        """
        Return the most common color in the grid.

        Args:
            grid (np.ndarray): The grid array containing color values.

        Returns:
            int: The most common color.
        """
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        return int(np.argmax(counts))

    def leastcolor(self, grid: np.ndarray) -> int:
        """
        Return the least common color in the grid.

        Args:
            grid (np.ndarray): The grid array containing color values.

        Returns:
            int: The least common color.
        """
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        # Replace zero counts with a big number to avoid selecting a color that is not present.
        counts[counts == 0] = self.big_number
        return int(np.argmin(counts))

    def rankcolor(self, grid: np.ndarray, rank: int) -> int:
        """
        Return the rank-th most common color in the grid.

        Args:
            grid (np.ndarray): The grid array.
            rank (int): Rank (0-based) to select the color.

        Returns:
            int: The color corresponding to the given rank.

        Raises:
            ValueError: If rank is not a non-negative integer.
        """
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("Rank must be a non-negative integer.")

        # Get unique colors and their occurrence counts.
        unique_colors, counts = np.unique(grid, return_counts=True)
        # Sort indices by counts in descending order.
        sorted_indices = np.argsort(-counts)

        if rank < len(unique_colors):
            return int(unique_colors[sorted_indices[rank]])
        else:
            # If rank exceeds available unique colors, return the least common among those present.
            last_nonzero_index = sorted_indices[-1]
            return int(unique_colors[last_nonzero_index])

    def rank_largest_shape_color_nodiag(self, grid: np.ndarray, rank: int) -> int:
        """
        Return the color of the rank-th largest connected shape in the grid (without diagonal connectivity).

        Args:
            grid (np.ndarray): The grid array containing color values.
            rank (int): Rank (0-based) for the largest shape.

        Returns:
            int: The color corresponding to the rank-th largest shape.

        Raises:
            ValueError: If rank is not a non-negative integer or if the grid is empty or non-integer.
        """
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("Rank must be a non-negative integer.")
        if not np.issubdtype(grid.dtype, np.integer) or grid.size == 0:
            raise ValueError("Grid must be a non-empty array of integers.")

        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)
        # Array to store the size of the largest connected shape for each unique color.
        dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)

        for i, color in enumerate(unique_colors):
            color_mask = grid == color
            # Label connected regions in the binary mask.
            labeled_grid, _ = label(color_mask.astype(int))
            # Get counts for each label; ignore the background (label 0).
            _, counts = np.unique(labeled_grid, return_counts=True)
            counts = counts[1:]  # Exclude background
            dimension_of_biggest_shape[i] = np.max(counts) if counts.size > 0 else 0

        # Sort colors by the size of their largest connected component in descending order.
        sorted_indices = np.argsort(-dimension_of_biggest_shape)
        if rank >= len(sorted_indices):
            # If rank is out-of-bounds, return the smallest non-zero shape or the first color if none.
            smallest_nonzero_index = np.argmin(
                np.where(dimension_of_biggest_shape > 0, dimension_of_biggest_shape, np.inf)
            )
            return int(unique_colors[smallest_nonzero_index]) if dimension_of_biggest_shape[smallest_nonzero_index] > 0 else int(unique_colors[0])

        return int(unique_colors[sorted_indices[rank]])

    def rank_largest_shape_color_diag(self, grid: np.ndarray, rank: int) -> int:
        """
        Return the color of the rank-th largest connected shape in the grid (considering diagonal connectivity).

        NOTE: This method is deprecated. Use rank_largest_shape_color_nodiag instead.

        Args:
            grid (np.ndarray): The grid array containing color values.
            rank (int): Rank (0-based) for the largest shape.

        Raises:
            DeprecationWarning: Always raised since this method is deprecated.
        """
        raise DeprecationWarning("This method is deprecated. Use rank_largest_shape_color_nodiag instead.")

        # Deprecated implementation (kept for reference):
        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)

        if not isinstance(rank, int) or rank < 0:
            raise ValueError("Rank must be a non-negative integer.")

        dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)
        # Define 8-connectivity (includes diagonals) for labeling.
        connectivity = np.ones((3, 3), dtype=int)

        for i, color in enumerate(unique_colors):
            copied_grid = np.copy(grid)
            color_mask = copied_grid == color
            copied_grid[color_mask] = 1
            copied_grid[~color_mask] = 0
            labeled_grid, _ = label(copied_grid, structure=connectivity)
            unique_labels, counts = np.unique(labeled_grid, return_counts=True)
            unique_labels, counts = unique_labels[1:], counts[1:]
            biggest_count = np.max(counts) if counts.size > 0 else 0
            dimension_of_biggest_shape[i] = biggest_count

        sorted_indices = np.argsort(-dimension_of_biggest_shape)
        if rank >= len(sorted_indices):
            non_zero_sizes = dimension_of_biggest_shape[dimension_of_biggest_shape > 0]
            if non_zero_sizes.size == 0:
                return 0
            smallest_nonzero_index = np.argmin(
                dimension_of_biggest_shape + (dimension_of_biggest_shape == 0) * np.max(dimension_of_biggest_shape)
            )
            return int(unique_colors[smallest_nonzero_index])

        index = sorted_indices[rank]
        return int(unique_colors[index])

    def color_number(self, grid: np.ndarray, color: int) -> int:
        """
        Return the given color if it is not present in the grid; otherwise, return an invalid color indicator.

        Args:
            grid (np.ndarray): The grid array.
            color (int): The color number to check.

        Returns:
            int: The color number if it is not in the grid, otherwise the invalid color value.
        """
        if not check_integer(color, 0, self.num_colors):
            return self.invalid_color
        unique_colors = np.unique(grid)
        if color in unique_colors:
            return self.invalid_color
        return color
