#%%
import numpy as np
import numba as nb
import time

# --------------------------------------------------------------------
# Transformation Utility Functions (Numba Version)
# --------------------------------------------------------------------

@nb.njit
def create_grid3d(grid, selection):
    """
    Create a 3D array by stacking the 2D grid a number of times equal to
    the number of layers in the selection. If grid is already 3D, return it.
    """
    if grid.ndim == 3:
        return grid
    num_selections = selection.shape[0]
    nrows = grid.shape[0]
    ncols = grid.shape[1]
    grid_3d = np.empty((num_selections, nrows, ncols), dtype=grid.dtype)
    for i in range(num_selections):
        for r in range(nrows):
            for c in range(ncols):
                grid_3d[i, r, c] = grid[r, c]
    return grid_3d

@nb.njit
def find_bounding_rectangle(input_array):
    """
    For each 2D slice in the 3D boolean array, compute the smallest bounding
    rectangle (all True) and return a 3D boolean array where, in each slice,
    only the bounding rectangle is True.
    """
    d, rows, cols = input_array.shape
    output_array = np.zeros((d, rows, cols), dtype=np.bool_)
    for k in range(d):
        min_row = rows
        max_row = -1
        min_col = cols
        max_col = -1
        for i in range(rows):
            for j in range(cols):
                if input_array[k, i, j]:
                    if i < min_row:
                        min_row = i
                    if i > max_row:
                        max_row = i
                    if j < min_col:
                        min_col = j
                    if j > max_col:
                        max_col = j
        if max_row >= min_row and max_col >= min_col:
            for i in range(min_row, max_row + 1):
                for j in range(min_col, max_col + 1):
                    output_array[k, i, j] = True
    return output_array

@nb.njit
def find_bounding_square(mask):
    """
    For each 2D slice in the 3D boolean array, find the smallest square
    that covers all True pixels. The square’s side length equals the larger
    of the rectangle’s height and width. The result is returned as a 3D array
    of int64 (with 1’s in the bounding square and 0’s elsewhere).
    """
    d, num_rows, num_cols = mask.shape
    bounding_masks = np.zeros((d, num_rows, num_cols), dtype=np.int64)
    for k in range(d):
        min_row = num_rows
        max_row = -1
        min_col = num_cols
        max_col = -1
        for i in range(num_rows):
            for j in range(num_cols):
                if mask[k, i, j]:
                    if i < min_row:
                        min_row = i
                    if i > max_row:
                        max_row = i
                    if j < min_col:
                        min_col = j
                    if j > max_col:
                        max_col = j
        if max_row == -1:
            # No True pixels found: leave slice as zeros.
            continue
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        side_length = height if height > width else width
        row_center = (min_row + max_row) // 2
        col_center = (min_col + max_col) // 2
        row_start_new = row_center - side_length // 2
        col_start_new = col_center - side_length // 2
        if row_start_new < 0:
            row_start_new = 0
        if col_start_new < 0:
            col_start_new = 0
        row_end_new = row_start_new + side_length
        col_end_new = col_start_new + side_length
        if row_end_new > num_rows:
            row_end_new = num_rows
            row_start_new = row_end_new - side_length
            if row_start_new < 0:
                row_start_new = 0
        if col_end_new > num_cols:
            col_end_new = num_cols
            col_start_new = col_end_new - side_length
            if col_start_new < 0:
                col_start_new = 0
        for i in range(row_start_new, row_end_new):
            for j in range(col_start_new, col_end_new):
                bounding_masks[k, i, j] = 1
    return bounding_masks

@nb.njit
def center_of_mass(bool_array):
    """
    Calculate the integer center of mass of the True values in a 2D boolean array.
    Returns a tuple (com_row, com_col).
    """
    rows, cols = bool_array.shape
    total = 0.0
    sum_i = 0.0
    sum_j = 0.0
    for i in range(rows):
        for j in range(cols):
            if bool_array[i, j]:
                total += 1.0
                sum_i += i
                sum_j += j
    if total == 0:
        return (0, 0)
    com_i = int(round(sum_i / total))
    com_j = int(round(sum_j / total))
    return (com_i, com_j)

@nb.njit
def vectorized_center_of_mass(selection):
    """
    For each 2D slice in a 3D boolean array, compute its center of mass.
    Returns a 2D array of shape (d, 2) with integer center indices.
    """
    d, rows, cols = selection.shape
    result = np.empty((d, 2), dtype=np.int64)
    for k in range(d):
        com = center_of_mass(selection[k])
        result[k, 0] = com[0]
        result[k, 1] = com[1]
    return result

@nb.njit
def missing_integer(grid):
    """
    Given a grid of integers in the range 0 to 9, return an integer in [0,9]
    that is not present in the grid. If all values exist, return 0.
    """
    found = np.zeros(10, dtype=np.bool_)
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            if val >= 0 and val < 10:
                found[val] = True
    for i in range(10):
        if not found[i]:
            return i
    return 0

# --------------------------------------------------------------------
# Testing Functions
# --------------------------------------------------------------------

def test_utilities():
    # Create a random grid of integers and a corresponding boolean mask.
    grid = np.random.randint(0, 10, size=(100, 100)).astype(np.int64)
    # For testing the transformation utilities, we simulate a selection mask.
    # For example, let the mask be True where grid values are even.
    selection_mask = (grid % 2) == 0
    # Convert the 2D mask into 3D by stacking one layer.
    selection_3d = selection_mask[np.newaxis, ...]
    
    grid3d = create_grid3d(grid, selection_3d)
    print("create_grid3d shape:", grid3d.shape)
    
    bounding_rect = find_bounding_rectangle(selection_3d)
    print("find_bounding_rectangle shape:", bounding_rect.shape)
    
    bounding_sq = find_bounding_square(selection_3d)
    print("find_bounding_square shape:", bounding_sq.shape)
    
    com = center_of_mass(selection_mask)
    print("center_of_mass of selection (2D):", com)
    
    vcom = vectorized_center_of_mass(selection_3d)
    print("vectorized_center_of_mass (3D):", vcom)
    
    missing = missing_integer(grid)
    print("missing_integer:", missing)

def run_time_tests():
    # Create a large random grid (e.g., 1000x1000) and a corresponding selection mask.
    grid = np.random.randint(0, 10, size=(1000, 1000)).astype(np.int64)
    selection_mask = (grid % 2) == 0
    selection_3d = selection_mask[np.newaxis, ...]
    iterations = 10
    
    start = time.time()
    for _ in range(iterations):
        _ = create_grid3d(grid, selection_3d)
    end = time.time()
    print("create_grid3d average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = find_bounding_rectangle(selection_3d)
    end = time.time()
    print("find_bounding_rectangle average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = find_bounding_square(selection_3d)
    end = time.time()
    print("find_bounding_square average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = center_of_mass(selection_mask)
    end = time.time()
    print("center_of_mass average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = vectorized_center_of_mass(selection_3d)
    end = time.time()
    print("vectorized_center_of_mass average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = missing_integer(grid)
    end = time.time()
    print("missing_integer average time: {:.6f} s".format((end - start) / iterations))

if __name__ == '__main__':
    print("=== Testing Utility Functions Numba===")
    test_utilities()
    print("\n=== Running Time Tests Numba===")
    run_time_tests()





















#%%

import numpy as np
from scipy.ndimage import find_objects
from scipy.ndimage import center_of_mass as scipy_com
from skimage.measure import regionprops

# Implemented utility methods:
# - create_grid3d(grid, selection): Add an additional dimension to the grid by stacking it.
# - find_bounding_rectangle(mask): Find the smallest bounding rectangle around non-zero regions in a binary mask.
# - find_bounding_square(mask): Find the smallest bounding square around non-zero regions in a binary mask.

def create_grid3d(grid, selection):
        #do not add an additional dimension if already 3D
        if len(grid.shape) == 3:
            return grid
        num_selections = selection.shape[0]
        grid_3d = np.stack([grid] * num_selections, axis=0)
        return grid_3d

def find_bounding_rectangle(input_array):
    """
    For each 2D slice in the 3D boolean array, calculate the bounding rectangle of `True` values
    and set the bounding rectangle to `True` in the output 3D array.

    Parameters:
    input_array (numpy.ndarray): 3D boolean array (d, rows, cols)

    Returns:
    numpy.ndarray: 3D boolean array with bounding rectangles of `True` values.
    """
    d, rows, cols = input_array.shape
    output_array = np.zeros_like(input_array, dtype=bool)

    for i in range(d):
        slice_2d = input_array[i]
        
        # Calculate the bounding box
        props = regionprops(slice_2d.astype(int))
        if props:
            min_row, min_col, max_row, max_col = props[0].bbox
            output_array[i, min_row:max_row, min_col:max_col] = True

    return output_array

def find_bounding_square(mask):
    """
    Return a mask of the same shape as the input, with bounding squares
    around non-zero regions set to 1 for each layer. Ensures the square
    is symmetric and aligned, truncating excess non-zero regions if necessary.
    """
    d, num_rows, num_cols = mask.shape
    bounding_masks = np.zeros_like(mask, dtype=int)

    for i in range(d):
        # Find bounding slices of the non-zero region
        i_th_slice = mask[i]
        slices = find_objects(i_th_slice)

        if slices:  # If there are non-zero regions
            bounding_box = slices[0]  # Assuming a single connected component
            row_start, row_end = bounding_box[0].start, bounding_box[0].stop
            col_start, col_end = bounding_box[1].start, bounding_box[1].stop

            # Calculate the size of the bounding box
            height = row_end - row_start
            width = col_end - col_start

            # Determine the side length of the square (maximum of height and width)
            side_length = max(height, width)

            # Center the square around the rectangle
            row_center = (row_start + row_end) // 2
            col_center = (col_start + col_end) // 2

            # Adjust row bounds symmetrically
            row_start_new = max(0, row_center - side_length // 2)
            row_end_new = min(num_rows, row_start_new + side_length)
            row_start_new = max(0, row_end_new - side_length)  # Adjust start if end exceeds bounds

            # Adjust column bounds symmetrically
            col_start_new = max(0, col_center - side_length // 2)
            col_end_new = min(num_cols, col_start_new + side_length)
            col_start_new = max(0, col_end_new - side_length)  # Adjust start if end exceeds bounds

            # Set the bounding square region to 1 in the result mask
            bounding_masks[i, row_start_new:row_end_new, col_start_new:col_end_new] = 1

    return bounding_masks

def center_of_mass(bool_array):
    """
    Calculate the integer indices of the center of mass of the `True` values in a NumPy boolean array.
    
    Parameters:
        bool_array (numpy.ndarray): A boolean array.
        
    Returns:
        tuple: A tuple of integers representing the center of mass indices along each axis.
    """
    # Ensure input is a numpy array
    bool_array = np.asarray(bool_array)
    # Calculate the center of mass as the mean of these indices
    center = scipy_com(bool_array)
    # Replace NaN with 0 and convert to integers
    center = tuple(int(np.nan_to_num(c, nan=0)) for c in center)
    return center

def vectorized_center_of_mass(selection):
     '''
     Caluclate the integer indeces of the center of mass of the true values in the selection tensor

     Parameters:
        selection (numpy.ndarray): A boolean array in 3 dimensions
    
    Returns:
        tuple: A tuple of integers representing the center of mass indices along each axis.
     '''
     depth, rows, cols = selection.shape
     total_weight = np.sum(selection, axis=(1, 2), keepdims=True)
     row_indices = np.arange(rows).reshape(1, rows, 1)
     weighted_rows = selection * row_indices
     sum_weighted_rows = np.sum(weighted_rows, axis=(1, 2), keepdims=True)
     com = np.round(np.where(total_weight > 0, sum_weighted_rows / total_weight, 0)).astype(int)
     return com

def missing_integer(grid: np.ndarray) -> int:
    """
    Takes a grid of integers (0 to 9) and returns an integer from 0 to 9 
    that is not present in the grid. If all integers are present, returns 0.
    """
    if not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("Grid must contain integers only.")

    # Find unique values in the grid
    unique_values = np.unique(grid)

    # Create a set of integers from 0 to 9
    all_integers = set(range(10))

    # Find missing integers
    missing_values = all_integers - set(unique_values)

    # Return the first missing value, or 0 if none are missing
    return min(missing_values) if missing_values else 0


# --------------------------------------------------------------------
# Testing Functions for the NumPy Code
# --------------------------------------------------------------------

def test_utilities():
    print("=== Testing Utility Functions Numpy ===")
    # Create a random grid (100x100) of integers 0-9.
    grid = np.random.randint(0, 10, size=(100, 100)).astype(np.int64)
    # Create a boolean selection mask; for example, select even numbers.
    selection_mask = (grid % 2) == 0
    # Convert 2D mask to 3D by adding one layer.
    selection_3d = selection_mask[np.newaxis, ...]
    
    # Test create_grid3d.
    grid3d = create_grid3d(grid, selection_3d)
    print("create_grid3d shape:", grid3d.shape)
    
    # Test find_bounding_rectangle.
    bounding_rect = find_bounding_rectangle(selection_3d)
    print("find_bounding_rectangle shape:", bounding_rect.shape)
    
    # Test find_bounding_square.
    bounding_sq = find_bounding_square(selection_3d)
    print("find_bounding_square shape:", bounding_sq.shape)
    
    # Test center_of_mass on the 2D mask.
    com = center_of_mass(selection_mask)
    print("center_of_mass (2D):", com)
    
    # Test vectorized_center_of_mass on the 3D mask.
    vcom = vectorized_center_of_mass(selection_3d)
    print("vectorized_center_of_mass (3D):", vcom)
    
    # Test missing_integer.
    missing = missing_integer(grid)
    print("missing_integer:", missing)

def run_time_tests():
    print("\n=== Running Time Tests Numpy===")
    # Create a large random grid (e.g., 1000x1000) of integers.
    grid = np.random.randint(0, 10, size=(1000, 1000)).astype(np.int64)
    # Create a boolean selection mask; for example, select even numbers.
    selection_mask = (grid % 2) == 0
    selection_3d = selection_mask[np.newaxis, ...]
    iterations = 10
    
    start = time.time()
    for _ in range(iterations):
        _ = create_grid3d(grid, selection_3d)
    end = time.time()
    print("create_grid3d average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = find_bounding_rectangle(selection_3d)
    end = time.time()
    print("find_bounding_rectangle average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = find_bounding_square(selection_3d)
    end = time.time()
    print("find_bounding_square average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = center_of_mass(selection_mask)
    end = time.time()
    print("center_of_mass average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = vectorized_center_of_mass(selection_3d)
    end = time.time()
    print("vectorized_center_of_mass average time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
        _ = missing_integer(grid)
    end = time.time()
    print("missing_integer average time: {:.6f} s".format((end - start) / iterations))

if __name__ == '__main__':
    test_utilities()
    run_time_tests()