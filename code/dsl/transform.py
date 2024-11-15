
import numpy as np
import matplotlib.pyplot as plt
from dsl.utilities.checks import check_axis, check_num_rotations
from scipy.ndimage import find_objects

# Implemented methods:
# - flip(grid, selection, axis): Flip the grid along the specified axis.
# - delete(grid, selection): Set the value of the selected cells to 0.

def find_bounding_rectangle(mask):
    """
    Find the smallest bounding rectangle around non-zero regions in a binary mask.
    """
    d, num_rows, num_cols = mask.shape
    for i in range(d):
        # Find bounding slices of the non-zero region
        i_th_slice = mask[i]
        bounding_box = find_objects(i_th_slice)[0] # This assumes a single connected component
        i_th_slice[bounding_box] = True    
    return mask

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

class Transformer:
    def __init__(self):
        pass

    def create_grid3d(self, grid, selection):
        num_selections = selection.shape[0]
        grid_3d = np.stack([grid] * num_selections, axis=0)
        return grid_3d

    def flip(self, grid, selection, axis):
        """
        Flip the grid along the specified axis.
        """
        grid_3d = self.create_grid3d(grid, selection) # Add an additional dimension to the grid by stacking it
        if check_axis(axis) == False:
            return grid_3d
        axis += 1 # Increas axis by 1 to account for the additional dimension
        new_selection = find_bounding_rectangle(selection) # Find the bounding rectangle around the selection for each slice
        flipped_selection = np.flip(new_selection, axis=axis) # Flip the selection along the specified axis
        grid_3d[new_selection] = np.flip(grid_3d, axis=axis)[flipped_selection] # Flip the bounding rectangle along the specified axis
        return grid_3d
    
    def delete(self, grid, selection):
        """
        Set the value of the selected cells to 0.
        """
        grid_3d = self.create_grid3d(grid, selection)
        grid_3d[selection] = 0
        return grid_3d
    
    def rotate(self, grid, selection, num_rotations):
        """
        Rotate the selected cells 90 degrees n times counterclockwise.
        """
        grid_3d = self.create_grid3d(grid, selection)
        if check_num_rotations(num_rotations) == False:
            return grid_3d
        print(selection)
        new_selection = find_bounding_square(selection)
        rotated_selection = np.rot90(new_selection, num_rotations)
        grid_3d[new_selection] = np.rot90(grid_3d, num_rotations)[rotated_selection]
        return grid_3d
    
    def color(self, grid, selection, color_selection_method, color_selection_param):
        """
        Color the selected cells using the specified color selection method.
        """
        # TODO: Select the coloring method from the ColorSelector class, then color the selected cells.
        return None


def find_bounding_square(array, mask):
    """
    Find the smallest square that fully encloses the non-zero regions in a binary mask.
    
    Parameters:
    - array: The 2D array for which the bounding square needs to be found (not directly used here, 
             but can be useful depending on the context).
    - mask:  A 2D binary mask (same shape as array) where non-zero values represent the region of interest.
    
    Returns:
    - (i_min, j_min, i_max, j_max): Coordinates of the top-left (i_min, j_min) and 
                                    bottom-right (i_max, j_max) corners of the smallest enclosing square.
    """

    # Find the smallest bounding rectangle around the region of interest
    i_min, j_min, i_max, j_max = find_bounding_rectangle(array, mask)

    # Calculate the size of the smallest square that can fully enclose the rectangle
    square_size = max(i_max - i_min + 1, j_max - j_min + 1)

    # Return the coordinates of the top-left and bottom-right corners of the square
    return i_min, j_min, i_min + square_size - 1, j_min + square_size - 1

def rotate(array, mask, n, display=False):
    """
    Rotate the input array 90 degrees n times counterclockwise.

    Parameters:
    - array: The input array to be rotated.
    - mask:  A 2D binary mask (same shape as array) where non-zero values represent the region of interest.
    - n: Number of 90-degree rotations to perform.

    Returns:
    - Rotated NumPy array.
    """

    i_min, j_min, i_max, j_max = find_bounding_square(array, mask)

    array[i_min:i_max+1, j_min:j_max+1] = np.rot90(array[i_min:i_max+1, j_min:j_max+1], n)

    if display:
        plot_array(array)
    
    return array

def drag(array, mask, row_shift, col_shift, display=True):
    """
    Drag a masked region of the array by a specified relative position (shift) and return the updated array.
    
    Parameters:
    - array: 2D array representing the main grid.
    - mask: Binary mask (same shape as array) where non-zero values define the region to drag.
    - row_shift: Integer representing the shift in the row direction (vertical shift).
    - col_shift: Integer representing the shift in the column direction (horizontal shift).
    - display: Boolean, if True, a visualization of the updated array is displayed.
    
    Returns:
    - A modified copy of the array with the masked region shifted by the specified relative positions.
    - If the shift would cause the region to go out of bounds, the original array is returned unchanged.
    """

    # Find the bounding rectangle of the region of interest in the mask
    i_min, j_min, i_max, j_max = find_bounding_rectangle(array, mask)

    # Calculate the new bounding coordinates after the shift
    i_min_new, j_min_new, i_max_new, j_max_new = i_min + row_shift, j_min + col_shift, i_max + row_shift, j_max + col_shift

    # Check if the shifted region goes out of bounds of the array
    if i_min_new < 0 or j_min_new < 0 or i_max_new >= np.shape(array)[0] or j_max_new >= np.shape(array)[1]:
        return array  # Return the original array if the shift moves the region out of bounds

    # Get the indices of the masked region in the original array
    mask_row_indices, mask_col_indices = np.where(mask)

    # Compute new indices for the shifted region
    new_row_indices = mask_row_indices + row_shift
    new_col_indices = mask_col_indices + col_shift

    # Create a copy of the original array to modify
    shifted_array = np.copy(array)

    # Clear the original masked region in the array
    shifted_array[mask_row_indices, mask_col_indices] = 0

    # Move the region to its new location in the array
    shifted_array[new_row_indices, new_col_indices] = array[mask_row_indices, mask_col_indices]

    # Optionally display the updated array
    if display:
        plot_array(shifted_array)

    # Return the modified array with the shifted region
    return shifted_array

def alt_drag(array, mask, row_shift, col_shift, display=True):
    """
    Drag a masked region of the array by a specified shift without clearing the original position, 
    and return the updated array.

    Parameters:
    - array: 2D array representing the main grid.
    - mask: Binary mask (same shape as array) where non-zero values define the region to drag.
    - row_shift: Integer representing the shift in the row direction (vertical shift).
    - col_shift: Integer representing the shift in the column direction (horizontal shift).
    - display: Boolean, if True, a visualization of the updated array is displayed.

    Returns:
    - The modified array with the masked region shifted by the specified shift, 
      but without clearing the original region.
    - If the shift would cause the region to go out of bounds, the original array is returned unchanged.
    """

    # Find the bounding rectangle of the region of interest in the mask
    i_min, j_min, i_max, j_max = find_bounding_rectangle(array, mask)

    # Calculate the new bounding coordinates after the shift
    i_min_new, j_min_new, i_max_new, j_max_new = i_min + row_shift, j_min + col_shift, i_max + row_shift, j_max + col_shift

    # Check if the shifted region goes out of bounds of the array
    if i_min_new < 0 or j_min_new < 0 or i_max_new >= np.shape(array)[0] or j_max_new >= np.shape(array)[1]:
        return array  # Return the original array if the shift moves the region out of bounds

    # Get the indices of the masked region in the original array
    mask_row_indices, mask_col_indices = np.where(mask)

    # Compute new indices for the shifted region
    new_row_indices = mask_row_indices + row_shift
    new_col_indices = mask_col_indices + col_shift

    # Move the region to its new location in the array without clearing the original location
    array[new_row_indices, new_col_indices] = array[mask_row_indices, mask_col_indices]

    # Optionally display the updated array
    if display:
        plot_array(array)

    # Return the modified array with the shifted region
    return array

def color_cell(array, pos, color):
    """
    Set the color of a cell at the given position.

    Parameters:
    - array: The input array.
    - pos: The position of the cell to be colored.
    - color: The color value to set (0-8).

    Returns:
    - Modified NumPy array with the specified cell colored.
    """
    array = convert_to_array(array)
    pos = convert_position(array, pos)
    color = convert_color(array, color)
    i, j = pos // np.shape(array)[1], pos % np.shape(array)[1]
    array[i, j] = color
    return array

def crop(array, pos1, pos2):
    """
    Crop the array to the rectangle defined by pos1 and pos2.

    Parameters:
    - array: The input array to be cropped.
    - pos1: The top-left position of the crop rectangle.
    - pos2: The bottom-right position of the crop rectangle.

    Returns:
    - Cropped NumPy array.
    """
    array = convert_to_array(array)
    pos1, pos2 = convert_position(array, pos1), convert_position(array, pos2)
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    i1, j1 = pos1 // np.shape(array)[1], pos1 % np.shape(array)[1]
    i2, j2 = pos2 // np.shape(array)[1], pos2 % np.shape(array)[1]
    array = array[i1:i2+1, j1:j2+1]
    return array