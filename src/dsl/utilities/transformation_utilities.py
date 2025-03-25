import numpy as np
# from scipy.ndimage import find_objects
# from scipy.ndimage import center_of_mass as scipy_com
# from skimage.measure import regionprops
import numba as nb

# Implemented utility methods:
# - create_grid3d(grid, selection): Add an additional dimension to the grid by stacking it.
# - find_bounding_rectangle(mask): Find the smallest bounding rectangle around non-zero regions in a binary mask.
# - find_bounding_square(mask): Find the smallest bounding square around non-zero regions in a binary mask.

@nb.njit
def create_grid3d(grid, selection):
    # #do not add an additional dimension if already 3D
    # if len(grid.shape) == 3:
    #     return grid
    # num_selections = selection.shape[0]
    # grid_3d = np.stack([grid] * num_selections, axis=0)
    # return grid_3d
    
    # Do not add an extra dimension if grid is already 3D.
    if grid.ndim == 3:
        return grid
    num_selections = selection.shape[0]
    # Stack grid num_selections times along new first dimension.
    grid_3d = np.empty((num_selections, grid.shape[0], grid.shape[1]), dtype=grid.dtype)
    for i in range(num_selections):
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                grid_3d[i, r, c] = grid[r, c]
    return grid_3d

@nb.njit
def find_bounding_rectangle(input_array):
    """
    For each 2D slice in the 3D boolean array, calculate the bounding rectangle of `True` values
    and set the bounding rectangle to `True` in the output 3D array.

    Parameters:
    input_array (numpy.ndarray): 3D boolean array (d, rows, cols)

    Returns:
    numpy.ndarray: 3D boolean array with bounding rectangles of `True` values.
    """
    # d, rows, cols = input_array.shape
    # output_array = np.zeros_like(input_array, dtype=bool)

    # for i in range(d):
    #     slice_2d = input_array[i]
        
    #     # Calculate the bounding box
    #     props = regionprops(slice_2d.astype(int))
    #     if props:
    #         min_row, min_col, max_row, max_col = props[0].bbox
    #         output_array[i, min_row:max_row, min_col:max_col] = True

    # return output_array
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

def find_bounding_square(mask):
    """
    Return a mask of the same shape as the input, with bounding squares
    around non-zero regions set to 1 for each layer. Ensures the square
    is symmetric and aligned, truncating excess non-zero regions if necessary.
    """
    # d, num_rows, num_cols = mask.shape
    # bounding_masks = np.zeros_like(mask, dtype=int)

    # for i in range(d):
    #     # Find bounding slices of the non-zero region
    #     i_th_slice = mask[i]
    #     slices = find_objects(i_th_slice)

    #     if slices:  # If there are non-zero regions
    #         bounding_box = slices[0]  # Assuming a single connected component
    #         row_start, row_end = bounding_box[0].start, bounding_box[0].stop
    #         col_start, col_end = bounding_box[1].start, bounding_box[1].stop

    #         # Calculate the size of the bounding box
    #         height = row_end - row_start
    #         width = col_end - col_start

    #         # Determine the side length of the square (maximum of height and width)
    #         side_length = max(height, width)

    #         # Center the square around the rectangle
    #         row_center = (row_start + row_end) // 2
    #         col_center = (col_start + col_end) // 2

    #         # Adjust row bounds symmetrically
    #         row_start_new = max(0, row_center - side_length // 2)
    #         row_end_new = min(num_rows, row_start_new + side_length)
    #         row_start_new = max(0, row_end_new - side_length)  # Adjust start if end exceeds bounds

    #         # Adjust column bounds symmetrically
    #         col_start_new = max(0, col_center - side_length // 2)
    #         col_end_new = min(num_cols, col_start_new + side_length)
    #         col_start_new = max(0, col_end_new - side_length)  # Adjust start if end exceeds bounds

    #         # Set the bounding square region to 1 in the result mask
    #         bounding_masks[i, row_start_new:row_end_new, col_start_new:col_end_new] = 1

    # return bounding_masks
    d, num_rows, num_cols = mask.shape
    bounding_masks = np.zeros_like(mask, dtype=np.int64)
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
            continue  # No True pixel found.
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        side_length = height if height > width else width
        # Center the square
        row_center = (min_row + max_row) // 2
        col_center = (min_col + max_col) // 2
        row_start_new = row_center - side_length // 2
        col_start_new = col_center - side_length // 2
        # Adjust if out of bounds.
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
    Calculate the integer indices of the center of mass of the `True` values in a NumPy boolean array.
    
    Parameters:
        bool_array (numpy.ndarray): A boolean array.
        
    Returns:
        tuple: A tuple of integers representing the center of mass indices along each axis.
    """
    # # Ensure input is a numpy array
    # bool_array = np.asarray(bool_array)
    # # Calculate the center of mass as the mean of these indices
    # center = scipy_com(bool_array)
    # # Replace NaN with 0 and convert to integers
    # center = tuple(int(np.nan_to_num(c, nan=0)) for c in center)
    # return center
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
    '''
    Calculate the integer indices of the center of mass of the true values in the selection tensor.

    Parameters:
        selection (numpy.ndarray): A boolean array in 3 dimensions.
    
    Returns:
        numpy.ndarray: A 2D array where each row contains the center of mass indices for a 2D slice.
    '''
    #  depth, rows, cols = selection.shape
    #  total_weight = np.sum(selection, axis=(1, 2), keepdims=True)
    #  row_indices = np.arange(rows).reshape(1, rows, 1)
    #  weighted_rows = selection * row_indices
    #  sum_weighted_rows = np.sum(weighted_rows, axis=(1, 2), keepdims=True)
    #  com = np.round(np.where(total_weight > 0, sum_weighted_rows / total_weight, 0)).astype(int)
    #  return com
    d, rows, cols = selection.shape
    result = np.empty((d, 2), dtype=np.int64)
    for k in range(d):
        com = center_of_mass(selection[k])
        result[k, 0] = com[0]
        result[k, 1] = com[1]
    return result

@nb.njit
def missing_integer(grid: np.ndarray) -> int:
    """
    Takes a grid of integers (0 to 9) and returns an integer from 0 to 9 
    that is not present in the grid. If all integers are present, returns 0.
    """
    # if not np.issubdtype(grid.dtype, np.integer):
    #     raise ValueError("Grid must contain integers only.")

    # # Find unique values in the grid
    # unique_values = np.unique(grid)

    # # Create a set of integers from 0 to 9
    # all_integers = set(range(10))

    # # Find missing integers
    # missing_values = all_integers - set(unique_values)

    # # Return the first missing value, or 0 if none are missing
    # return min(missing_values) if missing_values else 0
    def missing_integer(grid):
        # grid is assumed to contain integers.
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