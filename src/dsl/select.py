import numpy as np
# import matplotlib.pyplot as plt
# from dsl.utilities.checks import check_color, check_integer
# from skimage.segmentation import find_boundaries
# from scipy.ndimage import label, convolve
import numba as nb
from numba.typed import List

#Helper functions in numba that may be faster by replacing external functions

@nb.njit
def check_color_numba(color):
    # Assume valid colors are integers 0 to 9.
    return (color >= 0) and (color < 10)

@nb.njit
def check_integer_numba(val, min_val, max_val):
    # Check that val is between min_val and max_val (inclusive)
    return (val >= min_val) and (val <= max_val)

@nb.njit
def find_boundaries_numba(mask, mode_code):
    """
    Find boundaries in a binary mask.
    mode_code: 0 for "outer" (True pixel bordering a False neighbor)
               1 for "inner" (False pixel bordering a True neighbor)
    """
    rows, cols = mask.shape
    result = np.zeros((rows, cols), dtype=np.bool_)
    if mode_code == 0:  # outer: for each True pixel, if any 4-neighbor is False or outside, mark boundary.
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    is_bnd = False
                    if i == 0 or not mask[i-1, j]:
                        is_bnd = True
                    elif i == rows-1 or not mask[i+1, j]:
                        is_bnd = True
                    elif j == 0 or not mask[i, j-1]:
                        is_bnd = True
                    elif j == cols-1 or not mask[i, j+1]:
                        is_bnd = True
                    result[i, j] = is_bnd
        return result
    else:  # inner: for each False pixel, if any 4-neighbor is True, mark boundary.
        for i in range(rows):
            for j in range(cols):
                if not mask[i, j]:
                    is_bnd = False
                    if i > 0 and mask[i-1, j]:
                        is_bnd = True
                    elif i < rows-1 and mask[i+1, j]:
                        is_bnd = True
                    elif j > 0 and mask[i, j-1]:
                        is_bnd = True
                    elif j < cols-1 and mask[i, j+1]:
                        is_bnd = True
                    result[i, j] = is_bnd
        return result

@nb.njit
def label_numba(binary):
    """
    Simple 4-connected component labeling.
    Returns a labeled array and the number of features.
    """
    rows, cols = binary.shape
    labels = np.zeros((rows, cols), dtype=np.int64)
    current_label = 1
    # Preallocate a stack (maximum size = rows*cols)
    stack = np.empty((rows * cols, 2), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            if binary[i, j] and labels[i, j] == 0:
                sp = 0
                stack[sp, 0] = i
                stack[sp, 1] = j
                sp += 1
                labels[i, j] = current_label
                while sp > 0:
                    sp -= 1
                    x = stack[sp, 0]
                    y = stack[sp, 1]
                    # 4-neighbors
                    if x > 0 and binary[x-1, y] and labels[x-1, y] == 0:
                        labels[x-1, y] = current_label
                        stack[sp, 0] = x-1
                        stack[sp, 1] = y
                        sp += 1
                    if x < rows - 1 and binary[x+1, y] and labels[x+1, y] == 0:
                        labels[x+1, y] = current_label
                        stack[sp, 0] = x+1
                        stack[sp, 1] = y
                        sp += 1
                    if y > 0 and binary[x, y-1] and labels[x, y-1] == 0:
                        labels[x, y-1] = current_label
                        stack[sp, 0] = x
                        stack[sp, 1] = y-1
                        sp += 1
                    if y < cols - 1 and binary[x, y+1] and labels[x, y+1] == 0:
                        labels[x, y+1] = current_label
                        stack[sp, 0] = x
                        stack[sp, 1] = y+1
                        sp += 1
                current_label += 1
    return labels, current_label - 1

@nb.njit
def label_numba_diag(binary):
    """
    8-connected component labeling.
    """
    rows, cols = binary.shape
    labels = np.zeros((rows, cols), dtype=np.int64)
    current_label = 1
    stack = np.empty((rows * cols, 2), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            if binary[i, j] and labels[i, j] == 0:
                sp = 0
                stack[sp, 0] = i
                stack[sp, 1] = j
                sp += 1
                labels[i, j] = current_label
                while sp > 0:
                    sp -= 1
                    x = stack[sp, 0]
                    y = stack[sp, 1]
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            xx = x + dx
                            yy = y + dy
                            if xx >= 0 and xx < rows and yy >= 0 and yy < cols:
                                if binary[xx, yy] and labels[xx, yy] == 0:
                                    labels[xx, yy] = current_label
                                    stack[sp, 0] = xx
                                    stack[sp, 1] = yy
                                    sp += 1
                    # End inner loops
                current_label += 1
    return labels, current_label - 1

@nb.njit
def convolve_numba(image, kernel, cval):
    """
    2D convolution assuming mode "constant" for pixels outside the image.
    """
    rows, cols = image.shape
    krows, kcols = kernel.shape
    pad_r = krows // 2
    pad_c = kcols // 2
    result = np.empty((rows, cols), dtype=image.dtype)
    for i in range(rows):
        for j in range(cols):
            acc = 0
            for ki in range(krows):
                for kj in range(kcols):
                    ii = i + ki - pad_r
                    jj = j + kj - pad_c
                    val = cval
                    if ii >= 0 and ii < rows and jj >= 0 and jj < cols:
                        val = image[ii, jj]
                    acc += val * kernel[ki, kj]
            result[i, j] = acc
    return result



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
        # self.selection_vocabulary = {}  # Store the selection vocabulary.
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
        # if not check_color(color):
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # # Create a boolean mask where grid elements equal the target color.
        # mask = grid == color
        # n_rows, n_cols = grid.shape
        # mask = np.reshape(mask, (-1, n_rows, n_cols))

        # if np.sum(mask) == 0:
        #     raise Warning(f"Color {color} not found in the grid")
        # return mask
        
        # If color is not valid, return a 3D zero array.
        if not check_color_numba(color):
            return np.expand_dims(np.zeros_like(grid), axis=0)
        mask = grid == color
        n_rows, n_cols = grid.shape
        mask3d = mask.reshape((-1, n_rows, n_cols))
        if np.sum(mask3d) == 0:
            # In numba mode we cannot "raise" a Warning easily; return zero mask.
            return np.expand_dims(np.zeros_like(grid), axis=0)
        return mask3d

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
        # rows, cols = grid.shape
        # rectangles = []

        # # Validate height and width.
        # if not check_integer(height, self.minimum_geometry_size, rows):
        #     return np.expand_dims(np.zeros_like(grid), axis=0)
        # if not check_integer(width, self.minimum_geometry_size, cols):
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # # Obtain the boolean mask for the target color.
        # color_mask = self.select_color(grid, color)

        # # If no elements of the target color exist, return an empty mask.
        # if np.sum(color_mask) == 0:
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # # Remove the extra dimension.
        # color_mask = color_mask[0, :, :]

        # # Iterate over all possible starting positions for a rectangle.
        # for i in range(rows - height + 1):
        #     for j in range(cols - width + 1):
        #         sub_rect = color_mask[i : i + height, j : j + width]
        #         if np.all(sub_rect):
        #             # Create a mask with the rectangle set to True.
        #             rect_mask = np.zeros_like(color_mask, dtype=bool)
        #             rect_mask[i : i + height, j : j + width] = True
        #             rectangles.append(rect_mask)

        # # Stack all found rectangles into a 3D array.
        # if rectangles:
        #     result_3d = np.stack(rectangles, axis=0)
        # else:
        #     result_3d = np.zeros((0, *color_mask.shape), dtype=bool)
        # return result_3d
        rows, cols = grid.shape
        if not check_integer_numba(height, self.minimum_geometry_size, rows):
            return np.expand_dims(np.zeros_like(grid), axis=0)
        if not check_integer_numba(width, self.minimum_geometry_size, cols):
            return np.expand_dims(np.zeros_like(grid), axis=0)
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        color_mask = color_mask[0]  # remove extra dimension
        rect_list = List()
        for i in range(rows - height + 1):
            for j in range(cols - width + 1):
                sub_rect = color_mask[i:i+height, j:j+width]
                # np.all is supported if sub_rect is a NumPy array.
                if np.all(sub_rect):
                    rect_mask = np.zeros_like(color_mask, dtype=np.bool_)
                    for a in range(i, i+height):
                        for b in range(j, j+width):
                            rect_mask[a, b] = True
                    rect_list.append(rect_mask)
        if len(rect_list) > 0:
            # Stack the list along a new first dimension.
            result = np.stack(rect_list, axis=0)
        else:
            result = np.zeros((0, rows, cols), dtype=np.bool_)
        return result

    def select_connected_shapes(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select connected shapes (4-connectivity) in the grid corresponding to the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents a connected component.
        """
        # color_mask = self.select_color(grid, color)
        # if np.sum(color_mask) == 0:
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # # Remove the extra dimension.
        # color_mask = color_mask[0, :, :]

        # # Label connected components using 4-connectivity.
        # labeled_array, num_features = label(color_mask)

        # # Initialize a 3D array to store each connected component.
        # shape = (num_features, *color_mask.shape)
        # result_3d = np.zeros(shape, dtype=bool)

        # # Extract each connected component.
        # for i in range(1, num_features + 1):
        #     result_3d[i - 1] = (labeled_array == i)
        # return result_3d
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        color_mask = color_mask[0]
        labeled_array, num_features = label_numba(color_mask)
        result = np.zeros((num_features, color_mask.shape[0], color_mask.shape[1]), dtype=np.bool_)
        for i in range(1, num_features+1):
            for a in range(color_mask.shape[0]):
                for b in range(color_mask.shape[1]):
                    result[i-1, a, b] = (labeled_array[a, b] == i)
        return result

    def select_connected_shapes_diag(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select connected shapes (8-connectivity) in the grid corresponding to the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents a connected component.
        """
        # color_mask = self.select_color(grid, color)
        # if np.sum(color_mask) == 0:
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # # Remove the extra dimension.
        # color_mask = color_mask[0, :, :]

        # # Define an 8-connectivity structure.
        # structure = np.ones((3, 3), dtype=bool)
        # labeled_array, num_features = label(color_mask, structure)

        # # Initialize a 3D array to store each connected component.
        # shape = (num_features, *color_mask.shape)
        # result_3d = np.zeros(shape, dtype=bool)

        # for i in range(1, num_features + 1):
        #     result_3d[i - 1] = (labeled_array == i)
        # return result_3d
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        color_mask = color_mask[0]
        labeled_array, num_features = label_numba_diag(color_mask)
        result = np.zeros((num_features, color_mask.shape[0], color_mask.shape[1]), dtype=np.bool_)
        for i in range(1, num_features+1):
            for a in range(color_mask.shape[0]):
                for b in range(color_mask.shape[1]):
                    result[i-1, a, b] = (labeled_array[a, b] == i)
        return result

    def select_adjacent_to_color(self, grid: np.ndarray, color: int, points_of_contact: int) -> np.ndarray:
        """
        Find cells in the grid that are adjacent to the specified color with exactly the given number of contact points.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.
            points_of_contact (int): The required number of contact points (neighbors); valid range is 1 to 4.

        Returns:
            np.ndarray: A 3D boolean array where the first dimension is the selection mask.
        """
        # if not check_integer(points_of_contact, 1, 4):
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # nrows, ncols = grid.shape
        # if nrows == 0 or ncols == 0:
        #     return np.zeros((0, 0), dtype=bool)

        # color_mask = self.select_color(grid, color)
        # color_mask = color_mask[0, :, :]

        # # Define a kernel for counting non-diagonal neighbors.
        # kernel = np.array([[0, 1, 0],
        #                    [1, 0, 1],
        #                    [0, 1, 0]])
        # contact_count = convolve(color_mask.astype(int), kernel, mode="constant", cval=0)
        # selection_mask = (contact_count == points_of_contact) & ~color_mask
        # selection_mask = np.reshape(selection_mask, (-1, nrows, ncols))
        # return selection_mask
        if not check_integer_numba(points_of_contact, 1, 4):
            return np.expand_dims(np.zeros_like(grid), axis=0)
        nrows, ncols = grid.shape
        if nrows == 0 or ncols == 0:
            return np.zeros((0, 0), dtype=np.bool_)
        color_mask = self.select_color(grid, color)[0]
        # Define kernel for 4-connectivity.
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.int64)
        conv_result = convolve_numba(color_mask.astype(np.int64), kernel, 0)
        selection_mask = (conv_result == points_of_contact)
        # Exclude the pixels that already are of the color.
        for i in range(nrows):
            for j in range(ncols):
                if color_mask[i, j]:
                    selection_mask[i, j] = False
        return selection_mask.reshape((-1, nrows, ncols))

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
        # if not check_integer(points_of_contact, 1, 8):
        #     return np.expand_dims(np.zeros_like(grid), axis=0)

        # nrows, ncols = grid.shape
        # if nrows == 0 or ncols == 0:
        #     return np.zeros((0, 0), dtype=bool)

        # color_mask = self.select_color(grid, color)
        # color_mask = color_mask[0, :, :]

        # # Define a kernel for 8-connectivity.
        # kernel = np.ones((3, 3), dtype=bool)
        # contact_count = convolve(color_mask.astype(int), kernel, mode="constant", cval=0)
        # selection_mask = (contact_count == points_of_contact) & ~color_mask
        # selection_mask = np.reshape(selection_mask, (-1, nrows, ncols))
        # return selection_mask
        if not check_integer_numba(points_of_contact, 1, 8):
            return np.expand_dims(np.zeros_like(grid), axis=0)
        nrows, ncols = grid.shape
        if nrows == 0 or ncols == 0:
            return np.zeros((0, 0), dtype=np.bool_)
        color_mask = self.select_color(grid, color)[0]
        kernel = np.ones((3, 3), dtype=np.int64)
        conv_result = convolve_numba(color_mask.astype(np.int64), kernel, 0)
        selection_mask = (conv_result == points_of_contact)
        for i in range(nrows):
            for j in range(ncols):
                if color_mask[i, j]:
                    selection_mask[i, j] = False
        return selection_mask.reshape((-1, nrows, ncols))

    def select_outer_border(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the outer border of connected shapes with the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the outer border mask.
        """
        # color_separated_shapes = self.select_connected_shapes(grid, color)
        # for i in range(len(color_separated_shapes)):
        #     color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="outer")
        # return color_separated_shapes
        comps = self.select_connected_shapes(grid, color)
        n = comps.shape[0]
        for i in range(n):
            comps[i] = find_boundaries_numba(comps[i], 0)  # 0 = outer
        return comps

    def select_inner_border(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the inner border of connected shapes with the specified color.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the inner border mask.
        """
        # color_separated_shapes = self.select_connected_shapes(grid, color)
        # for i in range(len(color_separated_shapes)):
        #     color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="inner")
        # return color_separated_shapes
        comps = self.select_connected_shapes(grid, color)
        n = comps.shape[0]
        for i in range(n):
            comps[i] = find_boundaries_numba(comps[i], 1)  # 1 = inner
        return comps

    def select_outer_border_diag(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the outer border of connected shapes with the specified color using diagonal connectivity.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the outer border mask.
        """
        comps = self.select_connected_shapes_diag(grid, color)
        n = comps.shape[0]
        for i in range(n):
            comps[i] = find_boundaries_numba(comps[i], 0)
        return comps
        

    def select_inner_border_diag(self, grid: np.ndarray, color: int) -> np.ndarray:
        """
        Select the inner border of connected shapes with the specified color using diagonal connectivity.

        Args:
            grid (np.ndarray): A 2D grid array.
            color (int): The target color.

        Returns:
            np.ndarray: A 3D boolean array where each layer represents the inner border mask.
        """
        # color_separated_shapes = self.select_connected_shapes_diag(grid, color)
        # for i in range(len(color_separated_shapes)):
        #     color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode="inner")
        # return color_separated_shapes
        comps = self.select_connected_shapes_diag(grid, color)
        n = comps.shape[0]
        for i in range(n):
            comps[i] = find_boundaries_numba(comps[i], 1)
        return comps

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
        return np.ones((1, nrows, ncols), dtype=np.bool_)
