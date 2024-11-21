import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import regionprops

# Implemented utility methods:
# - create_grid3d(grid, selection): Add an additional dimension to the grid by stacking it.
# - find_bounding_rectangle(mask): Find the smallest bounding rectangle around non-zero regions in a binary mask.
# - find_bounding_square(mask): Find the smallest bounding square around non-zero regions in a binary mask.

def create_grid3d(grid, selection):
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
    Find the smallest bounding square around non-zero regions in a binary mask.
    """
    d, num_rows, num_cols = mask.shape
    for i in range(d):
        # Find bounding slices of the non-zero region
        i_th_slice = mask[i]
        slices = find_objects(i_th_slice)
        bounding_box = slices[0]  # Assuming a single connected component
        row_start, row_end = bounding_box[0].start, bounding_box[0].stop
        col_start, col_end = bounding_box[1].start, bounding_box[1].stop

        # Calculate the size of the bounding box
        height = row_end - row_start
        width = col_end - col_start

        # Determine the side length of the bounding square
        side_length = max(height, width)

        # Calculate new bounds to create a square
        row_center = (row_start + row_end) // 2
        col_center = (col_start + col_end) // 2

        row_start_new = max(0, row_center - side_length // 2)
        row_end_new = min(num_rows, row_start_new + side_length)
        row_start_new = max(0, row_end_new - side_length)  # Adjust start if end exceeds bounds

        col_start_new = max(0, col_center - side_length // 2)
        col_end_new = min(num_cols, col_start_new + side_length)
        col_start_new = max(0, col_end_new - side_length)  # Adjust start if end exceeds bounds

        # Set the bounding square region to True
        i_th_slice[row_start_new:row_end_new, col_start_new:col_end_new] = True
    return mask 

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
    
    # Get the indices of the True values
    indices = np.nonzero(bool_array)
    
    # Calculate the center of mass as the mean of these indices
    center = tuple(int(round(np.mean(axis))) for axis in indices)
    
    return center