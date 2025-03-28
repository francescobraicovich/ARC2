#%%
import numpy as np
import matplotlib.pyplot as plt
from dsl.utilities.checks import check_color, check_integer
from skimage.segmentation import find_boundaries
from scipy.ndimage import label, convolve
import numba as nb
from numba.typed import List
import time

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
def find_boundaries_numba_diag(mask, mode_code):
    """
    Find boundaries in a binary mask using 8-connected neighborhood.
    mode_code: 0 for "outer" (True pixel is on the border if any 8-neighbor is False or outside)
               1 for "inner" (False pixel is on the boundary if any 8-neighbor is True)
    """
    rows, cols = mask.shape
    result = np.zeros((rows, cols), dtype=np.bool_)
    if mode_code == 0:  # Outer boundary: for each True pixel, if any neighbor (8-connected) is False or outside, mark as boundary.
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    is_bnd = False
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = j + dj
                            # If neighbor is outside or False, mark boundary.
                            if ni < 0 or ni >= rows or nj < 0 or nj >= cols or not mask[ni, nj]:
                                is_bnd = True
                    result[i, j] = is_bnd
        return result
    else:  # Inner boundary: for each False pixel, if any neighbor (8-connected) is True, mark as boundary.
        for i in range(rows):
            for j in range(cols):
                if not mask[i, j]:
                    is_bnd = False
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = j + dj
                            if ni >= 0 and ni < rows and nj >= 0 and nj < cols and mask[ni, nj]:
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

#Functions to be used in the Selector class

# @nb.njit(parallel=True)
# def select_color_impl(grid, color):
#     # Ensure color is an integer
#     color = int(color)
#     n_rows = grid.shape[0]
#     n_cols = grid.shape[1]
#     # Use explicit comparison instead of "if not ..." to help numba
#     if check_color_numba(color) == False:
#         return np.zeros((1, n_rows, n_cols), dtype=grid.dtype)
#     mask = grid == color
#     mask3d = mask.reshape((1, n_rows, n_cols))
#     if np.sum(mask3d) == 0:
#         return np.zeros((1, n_rows, n_cols), dtype=grid.dtype)
#     return mask3d

# @nb.njit
# def select_color_impl(grid, color):
#     # Ensure color is an integer
#     color = int(color)
#     n_rows = grid.shape[0]
#     n_cols = grid.shape[1]
#     # Use a simple Boolean check
#     if not check_color_numba(color):
#         return np.zeros((1, n_rows, n_cols), dtype=grid.dtype)
#     mask = grid == color
#     mask3d = mask.reshape((1, n_rows, n_cols))
#     if np.count_nonzero(mask3d) == 0:
#         return np.zeros((1, n_rows, n_cols), dtype=grid.dtype)
#     return mask3d


@nb.njit(parallel=True)
def select_color_impl(grid, color):
    # Convert color to integer
    color = int(color)
    n_rows = grid.shape[0]
    n_cols = grid.shape[1]
    
    # If color is invalid, return a zero mask of booleans
    if not check_color_numba(color):
        return np.zeros((1, n_rows, n_cols), dtype=np.bool_)
    
    # Create a boolean mask array manually using parallel loops.
    mask = np.empty((n_rows, n_cols), dtype=np.bool_)
    for i in nb.prange(n_rows):
        for j in range(n_cols):
            mask[i, j] = (grid[i, j] == color)
    
    # Count the number of True entries in the mask
    cnt = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                cnt += 1
    
    # If no cell matches, return a zero mask.
    if cnt == 0:
        return np.zeros((1, n_rows, n_cols), dtype=np.bool_)
    
    # Reshape the mask to add the extra dimension and return it.
    return mask.reshape((1, n_rows, n_cols))


@nb.njit(parallel=True)
def select_rectangles_impl(grid, color, height, width, min_geom):
    rows, cols = grid.shape
    if not check_integer_numba(height, min_geom, rows) or not check_integer_numba(width, min_geom, cols):
        return np.expand_dims(np.zeros_like(grid), axis=0)
    color_mask = select_color_impl(grid, color)
    if np.sum(color_mask) == 0:
        return np.expand_dims(np.zeros_like(grid), axis=0)
    # Remove extra dimension.
    color_mask = color_mask[0]
    # We'll count the number of valid rectangles.
    valid_count = 0
    for i in range(rows - height + 1):
        for j in range(cols - width + 1):
            sub_rect = color_mask[i:i+height, j:j+width]
            if np.all(sub_rect):
                valid_count += 1
    # Preallocate result array.
    result = np.zeros((valid_count, rows, cols), dtype=np.bool_)
    idx = 0
    for i in range(rows - height + 1):
        for j in range(cols - width + 1):
            sub_rect = color_mask[i:i+height, j:j+width]
            if np.all(sub_rect):
                # Build rectangle mask.
                rect_mask = np.zeros_like(color_mask, dtype=np.bool_)
                for a in range(i, i+height):
                    for b in range(j, j+width):
                        rect_mask[a, b] = True
                result[idx] = rect_mask
                idx += 1
    return result

@nb.njit(parallel=True)
def select_connected_shapes_impl(grid: np.ndarray, color: int) -> np.ndarray:
    color_mask = select_color_impl(grid, color)
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

@nb.njit(parallel=True)
def select_connected_shapes_diag_impl(grid: np.ndarray, color: int) -> np.ndarray:
    color_mask = select_color_impl(grid, color)
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

@nb.njit(parallel=True)
def select_adjacent_to_color_impl(grid: np.ndarray, color: int, points_of_contact: int) -> np.ndarray:
    if not check_integer_numba(points_of_contact, 1, 4):
        return np.expand_dims(np.zeros_like(grid), axis=0)
    nrows, ncols = grid.shape
    if nrows == 0 or ncols == 0:
        return np.zeros((0, 0), dtype=np.bool_)
    color_mask = select_color_impl(grid, color)[0]
    # Define kernel for 4-connectivity.
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.int64)
    conv_result = convolve_numba(color_mask.astype(np.int64), kernel, 0)
    selection_mask = (conv_result == points_of_contact)
    for i in range(nrows):
        for j in range(ncols):
            if color_mask[i, j]:
                selection_mask[i, j] = False
    return selection_mask.reshape((-1, nrows, ncols))

@nb.njit(parallel=True)
def select_adjacent_to_color_diag_impl(grid, color, points_of_contact):
    if not check_integer_numba(points_of_contact, 1, 8):
        return np.expand_dims(np.zeros_like(grid), axis=0)
    nrows, ncols = grid.shape
    if nrows == 0 or ncols == 0:
        return np.zeros((0, 0), dtype=np.bool_)
    color_mask = select_color_impl(grid, color)[0]
    kernel = np.ones((3, 3), dtype=np.int64)
    conv_result = convolve_numba(color_mask.astype(np.int64), kernel, 0)
    selection_mask = (conv_result == points_of_contact)
    for i in range(nrows):
        for j in range(ncols):
            if color_mask[i, j]:
                selection_mask[i, j] = False
    return selection_mask.reshape((-1, nrows, ncols))

@nb.njit(parallel=True)
def select_outer_border_impl(grid, color):
    comps = select_connected_shapes_impl(grid, color)
    n = comps.shape[0]
    for i in nb.prange(n):
        comps[i] = find_boundaries_numba_diag(comps[i], 0)
    return comps

@nb.njit(parallel=True)
def select_inner_border_impl(grid, color):
    comps = select_connected_shapes_impl(grid, color)
    n = comps.shape[0]
    for i in nb.prange(n):
        comps[i] = find_boundaries_numba_diag(comps[i], 1)
    return comps

@nb.njit(parallel=True)
def select_outer_border_diag_impl(grid, color):
    comps = select_connected_shapes_diag_impl(grid, color)
    n = comps.shape[0]
    for i in nb.prange(n):
        comps[i] = find_boundaries_numba_diag(comps[i], 0)
    return comps
        
@nb.njit(parallel=True)
def select_inner_border_diag_impl(grid, color):
    comps = select_connected_shapes_diag_impl(grid, color)
    n = comps.shape[0]
    for i in nb.prange(n):
        comps[i] = find_boundaries_numba_diag(comps[i], 1)
    return comps

@nb.njit
def select_all_grid_impl(grid):
    nrows, ncols = grid.shape
    return np.ones((1, nrows, ncols), dtype=np.bool_)

class Selector:
    """
    A class for selecting elements from a grid using specific criteria.
    """
    def __init__(self):
        self.minimum_geometry_size = 2

    def select_color(self, grid, color):
        return select_color_impl(grid, color)

    def select_rectangles(self, grid, color, height, width):
        return select_rectangles_impl(grid, color, height, width, self.minimum_geometry_size)

    def select_connected_shapes(self, grid, color):
        return select_connected_shapes_impl(grid, color)

    def select_connected_shapes_diag(self, grid, color):
        return select_connected_shapes_diag_impl(grid, color)

    def select_adjacent_to_color(self, grid, color, points_of_contact):
        return select_adjacent_to_color_impl(grid, color, points_of_contact)

    def select_adjacent_to_color_diag(self, grid, color, points_of_contact):
        return select_adjacent_to_color_diag_impl(grid, color, points_of_contact)

    def select_outer_border(self, grid, color):
        return select_outer_border_impl(grid, color)

    def select_inner_border(self, grid, color):
        return select_inner_border_impl(grid, color)

    def select_outer_border_diag(self, grid, color):
        return select_outer_border_diag_impl(grid, color)

    def select_inner_border_diag(self, grid, color):
        return select_inner_border_diag_impl(grid, color)

    def select_all_grid(self, grid, color=None):
        return select_all_grid_impl(grid)

# --------------------------------------------------------------------
# Test block: Create a large random grid and time each function.
# --------------------------------------------------------------------
if __name__ == '__main__':
    # Create a random grid (e.g., 1000x1000) with colors 0..9.
    grid = np.random.randint(0, 10, size=(1000, 1000)).astype(np.int64)
    sel = Selector()
    iterations = 100

    # Warm-up (compile jit functions)
    _ = sel.select_color(grid, 5)
    _ = sel.select_rectangles(grid, 3, 10, 10)
    _ = sel.select_connected_shapes(grid, 3)
    _ = sel.select_connected_shapes_diag(grid, 3)
    _ = sel.select_adjacent_to_color(grid, 3, 2)
    _ = sel.select_adjacent_to_color_diag(grid, 3, 4)
    _ = sel.select_outer_border(grid, 3)
    _ = sel.select_inner_border(grid, 3)
    _ = sel.select_outer_border_diag(grid, 3)
    _ = sel.select_inner_border_diag(grid, 3)
    _ = sel.select_all_grid(grid)

    # Timing tests
    start = time.time()
    for _ in range(iterations):
        _ = sel.select_color(grid, 5)
    end = time.time()
    print("Numba select_color average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_rectangles(grid, 3, 10, 10)
    end = time.time()
    print("Numba select_rectangles average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_connected_shapes(grid, 3)
    end = time.time()
    print("Numba select_connected_shapes average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_connected_shapes_diag(grid, 3)
    end = time.time()
    print("Numba select_connected_shapes_diag average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_adjacent_to_color(grid, 3, 2)
    end = time.time()
    print("Numba select_adjacent_to_color average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_adjacent_to_color_diag(grid, 3, 4)
    end = time.time()
    print("Numba select_adjacent_to_color_diag average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_outer_border(grid, 3)
    end = time.time()
    print("Numba select_outer_border average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_inner_border(grid, 3)
    end = time.time()
    print("Numba select_inner_border average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_outer_border_diag(grid, 3)
    end = time.time()
    print("Numba select_outer_border_diag average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_inner_border_diag(grid, 3)
    end = time.time()
    print("Numba select_inner_border_diag average time: {:.6f} s".format((end - start) / iterations))

    start = time.time()
    for _ in range(iterations):
        _ = sel.select_all_grid(grid)
    end = time.time()
    print("Numba select_all_grid average time: {:.6f} s".format((end - start) / iterations))
