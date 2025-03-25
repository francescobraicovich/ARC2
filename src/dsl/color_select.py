#%%
import numpy as np
from dsl.utilities.checks import check_color_rank, check_integer
from scipy.ndimage import find_objects, label
import numba as nb


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


    @nb.njit
    def mostcolor(self, grid: np.ndarray) -> int:
        """
        Return the most common color in the grid.

        Args:
            grid (np.ndarray): The grid array containing color values.

        Returns:
            int: The most common color.
        """
        # values = grid.flatten()
        # counts = np.bincount(values, minlength=self.num_colors)
        # return int(np.argmax(counts))
        rows, cols = grid.shape
        counts = np.zeros(self.num_colors, dtype=np.int64)
        for i in range(rows):
            for j in range(cols):
                counts[grid[i, j]] += 1
        max_val = counts[0]
        max_color = 0
        for c in range(1, self.num_colors):
            if counts[c] > max_val:
                max_val = counts[c]
                max_color = c
        return max_color


    @nb.njit
    def leastcolor(grid: np.ndarray, num_colors: int, big_number: int) -> int:
        """
        Return the least common color in the grid.

        Args:
            grid (np.ndarray): The grid array containing color values.

        Returns:
            int: The least common color.
        """
        # values = grid.flatten()
        # counts = np.bincount(values, minlength=self.num_colors)
        # # Replace zero counts with a big number to avoid selecting a color that is not present.
        # counts[counts == 0] = self.big_number
        # return int(np.argmin(counts))
        rows, cols = grid.shape
        counts = np.zeros(num_colors, dtype=np.int64)
        for i in range(rows):
            for j in range(cols):
                counts[grid[i, j]] += 1
        # Replace zero counts with a big number.
        for c in range(num_colors):
            if counts[c] == 0:
                counts[c] = big_number
        min_val = counts[0]
        min_color = 0
        for c in range(1, num_colors):
            if counts[c] < min_val:
                min_val = counts[c]
                min_color = c
        return min_color


    @nb.njit
    def rankcolor(grid: np.ndarray, num_colors: int, rank: int, big_number: int) -> int:
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
        # if not isinstance(rank, int) or rank < 0:
        #     raise ValueError("Rank must be a non-negative integer.")

        # # Get unique colors and their occurrence counts.
        # unique_colors, counts = np.unique(grid, return_counts=True)
        # # Sort indices by counts in descending order.
        # sorted_indices = np.argsort(-counts)

        # if rank < len(unique_colors):
        #     return int(unique_colors[sorted_indices[rank]])
        # else:
        #     # If rank exceeds available unique colors, return the least common among those present.
        #     last_nonzero_index = sorted_indices[-1]
        #     return int(unique_colors[last_nonzero_index])
        rows, cols = grid.shape
        counts = np.zeros(num_colors, dtype=np.int64)
        for i in range(rows):
            for j in range(cols):
                counts[grid[i, j]] += 1
        # Replace zeros with big_number so that absent colors count as very high.
        for c in range(num_colors):
            if counts[c] == 0:
                counts[c] = big_number
        # Create an array of color indices.
        sorted_indices = np.empty(num_colors, dtype=np.int64)
        for i in range(num_colors):
            sorted_indices[i] = i
        # Simple selection sort: sort indices in descending order by counts.
        for i in range(num_colors):
            max_idx = i
            for j in range(i+1, num_colors):
                if counts[sorted_indices[j]] > counts[sorted_indices[max_idx]]:
                    max_idx = j
            # Swap positions
            temp = sorted_indices[i]
            sorted_indices[i] = sorted_indices[max_idx]
            sorted_indices[max_idx] = temp
        if rank < num_colors:
            return sorted_indices[rank]
        else:
            return sorted_indices[num_colors - 1]
        
    
    @staticmethod
    def rank_largest_shape_color_diag(grid: np.ndarray) -> int:
        """
        Return the color of the rank-th largest connected shape in the grid (considering diagonal connectivity).

        NOTE: This method is deprecated. Use rank_largest_shape_color_nodiag instead.

        Args:
            grid (np.ndarray): The grid array containing color values.
            rank (int): Rank (0-based) for the largest shape.
        """
        # raise DeprecationWarning("This method is deprecated. Use rank_largest_shape_color_nodiag instead.")

        # # Deprecated implementation (kept for reference):
        # unique_colors = np.unique(grid)
        # num_colors = len(unique_colors)

        # if not isinstance(rank, int) or rank < 0:
        #     raise ValueError("Rank must be a non-negative integer.")

        # dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)
        # # Define 8-connectivity (includes diagonals) for labeling.
        # connectivity = np.ones((3, 3), dtype=int)

        # for i, color in enumerate(unique_colors):
        #     copied_grid = np.copy(grid)
        #     color_mask = copied_grid == color
        #     copied_grid[color_mask] = 1
        #     copied_grid[~color_mask] = 0
        #     labeled_grid, _ = label(copied_grid, structure=connectivity)
        #     unique_labels, counts = np.unique(labeled_grid, return_counts=True)
        #     unique_labels, counts = unique_labels[1:], counts[1:]
        #     biggest_count = np.max(counts) if counts.size > 0 else 0
        #     dimension_of_biggest_shape[i] = biggest_count

        # sorted_indices = np.argsort(-dimension_of_biggest_shape)
        # if rank >= len(sorted_indices):
        #     non_zero_sizes = dimension_of_biggest_shape[dimension_of_biggest_shape > 0]
        #     if non_zero_sizes.size == 0:
        #         return 0
        #     smallest_nonzero_index = np.argmin(
        #         dimension_of_biggest_shape + (dimension_of_biggest_shape == 0) * np.max(dimension_of_biggest_shape)
        #     )
        #     return int(unique_colors[smallest_nonzero_index])

        # index = sorted_indices[rank]
        # return int(unique_colors[index])
        rows, cols = grid.shape
        visited = np.zeros((rows, cols), dtype=np.bool_)
        max_size = 0
        # Loop over each cell.
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] and not visited[i, j]:
                    size = 0
                    # Preallocate stack for DFS.
                    stack = np.empty((rows * cols, 2), dtype=np.int64)
                    sp = 0
                    stack[sp, 0] = i
                    stack[sp, 1] = j
                    sp += 1
                    visited[i, j] = True
                    while sp > 0:
                        sp -= 1
                        cur_i = stack[sp, 0]
                        cur_j = stack[sp, 1]
                        size += 1
                        # Check neighbors (4-connectivity)
                        if cur_i > 0 and grid[cur_i - 1, cur_j] and not visited[cur_i - 1, cur_j]:
                            visited[cur_i - 1, cur_j] = True
                            stack[sp, 0] = cur_i - 1
                            stack[sp, 1] = cur_j
                            sp += 1
                        if cur_i < rows - 1 and grid[cur_i + 1, cur_j] and not visited[cur_i + 1, cur_j]:
                            visited[cur_i + 1, cur_j] = True
                            stack[sp, 0] = cur_i + 1
                            stack[sp, 1] = cur_j
                            sp += 1
                        if cur_j > 0 and grid[cur_i, cur_j - 1] and not visited[cur_i, cur_j - 1]:
                            visited[cur_i, cur_j - 1] = True
                            stack[sp, 0] = cur_i
                            stack[sp, 1] = cur_j - 1
                            sp += 1
                        if cur_j < cols - 1 and grid[cur_i, cur_j + 1] and not visited[cur_i, cur_j + 1]:
                            visited[cur_i, cur_j + 1] = True
                            stack[sp, 0] = cur_i
                            stack[sp, 1] = cur_j + 1
                            sp += 1
                    if size > max_size:
                        max_size = size
        return max_size
        
        
    @nb.njit
    def rank_largest_shape_color_nodiag(self, grid: np.ndarray, num_colors: int, rank: int) -> int:
        """
        Return the color of the rank-th largest connected shape in the grid (without diagonal connectivity).

        Args:
            grid (np.ndarray): The grid array containing color values.
            num_colors (int): The number of unique colors in the grid.
            rank (int): Rank (0-based) for the largest shape.

        Returns:
            int: The color corresponding to the rank-th largest shape.

        Raises:
            ValueError: If rank is not a non-negative integer or if the grid is empty or non-integer.
        """
        # if not isinstance(rank, int) or rank < 0:
        #     raise ValueError("Rank must be a non-negative integer.")
        # if not np.issubdtype(grid.dtype, np.integer) or grid.size == 0:
        #     raise ValueError("Grid must be a non-empty array of integers.")

        # unique_colors = np.unique(grid)
        # num_colors = len(unique_colors)
        # # Array to store the size of the largest connected shape for each unique color.
        # dimension_of_biggest_shape = np.zeros(num_colors, dtype=int)

        # for i, color in enumerate(unique_colors):
        #     color_mask = grid == color
        #     # Label connected regions in the binary mask.
        #     labeled_grid, _ = label(color_mask.astype(int))
        #     # Get counts for each label; ignore the background (label 0).
        #     _, counts = np.unique(labeled_grid, return_counts=True)
        #     counts = counts[1:]  # Exclude background
        #     dimension_of_biggest_shape[i] = np.max(counts) if counts.size > 0 else 0

        # # Sort colors by the size of their largest connected component in descending order.
        # sorted_indices = np.argsort(-dimension_of_biggest_shape)
        # if rank >= len(sorted_indices):
        #     # If rank is out-of-bounds, return the smallest non-zero shape or the first color if none.
        #     smallest_nonzero_index = np.argmin(
        #         np.where(dimension_of_biggest_shape > 0, dimension_of_biggest_shape, np.inf)
        #     )
        #     return int(unique_colors[smallest_nonzero_index]) if dimension_of_biggest_shape[smallest_nonzero_index] > 0 else int(unique_colors[0])

        # return int(unique_colors[sorted_indices[rank]])
        rows, cols = grid.shape
        max_sizes = np.zeros(num_colors, dtype=np.int64)
        # For each possible color, compute its maximum connected component size.
        for color in range(num_colors):
            # Create binary mask for this color.
            mask = np.empty((rows, cols), dtype=np.bool_)
            for i in range(rows):
                for j in range(cols):
                    mask[i, j] = (grid[i, j] == color) 
            max_sizes[color] = np.sum(mask)  # Count the number of pixels of this color.
        # Create an array of color indices.
        sorted_indices = np.empty(num_colors, dtype=np.int64)
        for i in range(num_colors):
            sorted_indices[i] = i
        # Sort indices in descending order of max_sizes.
        for i in range(num_colors):
            max_idx = i
            for j in range(i + 1, num_colors):
                if max_sizes[sorted_indices[j]] > max_sizes[sorted_indices[max_idx]]:
                    max_idx = j
            temp = sorted_indices[i]
            sorted_indices[i] = sorted_indices[max_idx]
            sorted_indices[max_idx] = temp
        if rank < num_colors:
            return sorted_indices[rank]
        else:
            # If rank is out-of-bounds, return the color with the smallest nonzero size,
            # or 0 if no nonzero sizes exist.
            for i in range(num_colors):
                if max_sizes[sorted_indices[i]] > 0:
                    return sorted_indices[i]
            return 0


    @nb.njit
    def color_number(grid: np.ndarray, num_colors: int, color: int, invalid_color) -> int:
        """
        Return the given color if it is not present in the grid; otherwise, return an invalid color indicator.

        Args:
            grid (np.ndarray): The grid array.
            color (int): The color number to check.

        Returns:
            int: The color number if it is not in the grid, otherwise the invalid color value.
        """
        # if not check_integer(color, 0, self.num_colors):
        #     return self.invalid_color
        # unique_colors = np.unique(grid)
        # if color in unique_colors:
        #     return self.invalid_color
        # return color
        if color < 0 or color >= num_colors:
            return invalid_color
        rows, cols = grid.shape
        found = False
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == color:
                    found = True
                    break
            if found:
                break
        if found:
            return invalid_color
        else:
            return color
