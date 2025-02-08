import numpy as np
import matplotlib.pyplot as plt
from dsl.utilities.checks import check_color, check_integer
from skimage.segmentation import find_boundaries
from scipy.ndimage import label, convolve


class Selector:
    """
    A class for selecting elements from a grid based on specific criteria.

    Implemented methods:
      1. select_color: Select grid elements matching a specific color.
      2. select_rectangles: Extract rectangles of a specific color, height, and width.
      3. select_connected_shapes: Select connected shapes (4-connectivity) of a specific color.
      4. select_connected_shapes_diag: Select connected shapes (8-connectivity) of a specific color.
      5. select_adjacent_to_color: Select cells adjacent to a given color with a specified number of contact points.
      6. select_adjacent_to_color_diag: As above, considering diagonal connectivity.
      7. select_outer_border: Select the outer border of connected shapes with a specific color.
      8. select_inner_border: Select the inner border of connected shapes with a specific color.
      9. select_outer_border_diag: Select the outer border with diagonal connectivity.
     10. select_inner_border_diag: Select the inner border with diagonal connectivity.
     11. select_all_grid: Select the entire grid.
    """

    def __init__(self):
        """
        Initialize the Selector instance.
        """
        self.selection_vocabulary = {}  # Store the selection vocabulary.
        self.minimum_geometry_size = 2  # Minimum allowed size for geometry dimensions.

    def select_color(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select grid elements matching the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where the first dimension is the selection mask.

        Raises:
            Warning: If the specified color is not found in the grid.
        """
        if not check_color(color):
            return np.expand_dims(np.zeros_like(grid), axis=0)

        # Create a boolean mask where grid elements equal the target color.
        mask = grid == color
        n_rows, n_cols = grid.shape
        mask = np.reshape(mask, (-1, n_rows, n_cols))

        if np.sum(mask) == 0:
            raise Warning(f"Color {color} not found in the grid")
        return mask

    def select_rectangles(self, grid: np.ndarray, color: int, height: int, width: int) -> np.ndarray:
        """
        Extract all possible rectangles of a given height and width from the grid where all elements are True.

        This method returns a 3D boolean array where each layer corresponds to one valid rectangle.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color for rectangle extraction.
            height (int): Height of the rectangle.
            width (int): Width of the rectangle.

        Returns:
            np.ndarray: A 3D boolean array with each layer representing a rectangle mask.
        """
        rows, cols = grid.shape
        rectangles = []

        # Validate height and width.
        if not check_integer(height, self.minimum_geometry_size, rows):
            return np.expand_dims(np.zeros_like(grid), axis=0)
        if not check_integer(width, self.minimum_geometry_size, cols):
            return np.expand_dims(np.zeros_like(grid), axis=0)

        # Obtain the boolean mask for the target color.
        color_mask = self.select_color(grid, color)

        # If no elements of the target color exist, return an empty mask.
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)

        # Remove the extra dimension.
        color_mask = color_mask[0, :, :]

        # Iterate over all possible starting positions for a rectangle.
        for i in range(rows - height + 1):
            for j in range(cols - width + 1):
                sub_rect = color_mask[i : i + height, j : j + width]
                if np.all(sub_rect):
                    # Create a mask with the rectangle set to True.
                    rect_mask = np.zeros_like(color_mask, dtype=bool)
                    rect_mask[i : i + height, j : j + width] = True
                    rectangles.append(rect_mask)

        # Stack all found rectangles into a 3D array.
        if rectangles:
            result_3d = np.stack(rectangles, axis=0)
        else:
            result_3d = np.zeros((0, *color_mask.shape), dtype=bool)
        return result_3d

    def select_connected_shapes(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select connected shapes (4-connectivity) in the grid corresponding to the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents a connected component.
        """
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)

        # Remove the extra dimension.
        color_mask = color_mask[0, :, :]

        # Label connected components using 4-connectivity.
        labeled_array, num_features = label(color_mask)

        # Initialize a 3D array to store each connected component.
        shape = (num_features, *color_mask.shape)
        result_3d = np.zeros(shape, dtype=bool)

        # Extract each connected component.
        for i in range(1, num_features + 1):
            result_3d[i - 1] = (labeled_array == i)
        return result_3d

    def select_connected_shapes_diag(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select connected shapes (8-connectivity) in the grid corresponding to the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents a connected component.
        """
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)

        # Remove the extra dimension.
        color_mask = color_mask[0, :, :]

        # Define an 8-connectivity structure.
        structure = np.ones((3, 3), dtype=bool)
        labeled_array, num_features = label(color_mask, structure)

        # Initialize a 3D array to store each connected component.
        shape = (num_features, *color_mask.shape)
        result_3d = np.zeros(shape, dtype=bool)

        for i in range(1, num_features + 1):
            result_3d[i - 1] = (labeled_array == i)
        return result_3d

    def select_adjacent_to_color(
        self, grid: np.ndarray, color: int, points_of_contact: int
    ) -> np.ndarray:
        """
        Find cells in the grid that are adjacent to the specified color with exactly the given number of contact points.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.
            points_of_contact (int): The required number of contact points (neighbors); valid range is 1 to 4.

        Returns:
            np.ndarray: A 3D boolean array where the first dimension is the selection mask.
        """
        if not check_integer(points_of_contact, 1, 4):
            return np.expand_dims(np.zeros_like(grid), axis=0)

        nrows, ncols = grid.shape
        if nrows == 0 or ncols == 0:
            return np.zeros((0, 0), dtype=bool)

        color_mask = self.select_color(grid, color)
        color_mask = color_mask[0, :, :]

        # Define a kernel for counting non-diagonal neighbors.
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
        contact_count = convolve(color_mask.astype(int), kernel, mode="constant", cval=0)
        selection_mask = (contact_count == points_of_contact) & ~color_mask
        selection_mask = np.reshape(selection_mask, (-1, nrows, ncols))
        return selection_mask

    def select_adjacent_to_color_diag(
        self, grid: np.ndarray, color: int, points_of_contact: int
    ) -> np.ndarray:
        """
        Find cells in the grid that are adjacent to the specified color with exactly the given number of contact points,
        considering diagonal neighbors (8-connectivity).

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.
            points_of_contact (int): The required number of contact points (neighbors); valid range is 1 to 8.

        Returns:
            np.ndarray: A 3D boolean array where the first dimension is the selection mask.
        """
        if not check_integer(points_of_contact, 1, 8):
            return np.expand_dims(np.zeros_like(grid), axis=0)

        nrows, ncols = grid.shape
        if nrows == 0 or ncols == 0:
            return np.zeros((0, 0), dtype=bool)

        color_mask = self.select_color(grid, color)
        color_mask = color_mask[0, :, :]

        # Define a kernel for 8-connectivity.
        kernel = np.ones((3, 3), dtype=bool)
        contact_count = convolve(color_mask.astype(int), kernel, mode="constant", cval=0)
        selection_mask = (contact_count == points_of_contact) & ~color_mask
        selection_mask = np.reshape(selection_mask, (-1, nrows, ncols))
        return selection_mask

    def select_outer_border(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the outer border of connected shapes with the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the outer border mask.
        """
        color_separated_shapes = self.select_connected_shapes(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="outer")
        return color_separated_shapes

    def select_inner_border(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the inner border of connected shapes with the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the inner border mask.
        """
        color_separated_shapes = self.select_connected_shapes(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="inner")
        return color_separated_shapes

    def select_outer_border_diag(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the outer border of connected shapes with the specified color using diagonal connectivity.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the outer border mask.
        """
        color_separated_shapes = self.select_connected_shapes_diag(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="outer")
        return color_separated_shapes

    def select_inner_border_diag(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the inner border of connected shapes with the specified color using diagonal connectivity.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the inner border mask.
        """
        color_separated_shapes = self.select_connected_shapes_diag(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="inner")
        return color_separated_shapes

    def select_all_grid(self, grid: np.ndarray, color: int = None) -> np.ndarray:
        """
        Select the entire grid.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int, optional): An optional color parameter (ignored).

        Returns:
            np.ndarray: A 3D boolean array with one layer where all values are True.
        """
        nrows, ncols = grid.shape
        return np.ones((1, nrows, ncols), dtype=bool)
