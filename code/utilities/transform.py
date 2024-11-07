
import numpy as np
import matplotlib.pyplot as plt

def plot_array(array):
    """
    Plot the input array as an image.

    Parameters:
    - array: The input array to be plotted.
    """
    plt.imshow(array, cmap='inferno')
    plt.show()

def find_bounding_rectangle(array, mask):
    """
    Find the smallest bounding rectangle around non-zero regions in a binary mask.
    
    Parameters:
    - array: The 2D array for which the bounding rectangle needs to be found (not directly used here, 
             but can be useful depending on the context).
    - mask:  A 2D binary mask (same shape as array) where non-zero values represent the region of interest.
    
    Returns:
    - (i_min, j_min, i_max, j_max): Coordinates of the top-left (i_min, j_min) and 
                                    bottom-right (i_max, j_max) corners of the smallest bounding rectangle.
    """

    # Get the row (i) and column (j) indices where the mask is True (non-zero)
    row_indices, col_indices = np.where(mask)

    # Find the minimum and maximum row indices
    i_min, i_max = min(row_indices), max(row_indices)

    # Find the minimum and maximum column indices
    j_min, j_max = min(col_indices), max(col_indices)

    # Return the coordinates of the top-left and bottom-right corners of the bounding rectangle
    return i_min, j_min, i_max, j_max

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

def delete(array, mask, display=False):
    """
    Set the value of a cell at the given position to 0.

    Parameters:
    - array: The input array.
    - mask: The binary mask specifying the cells to be deleted.
    - pos: The position of the cell to be deleted.

    Returns:
    - Modified NumPy array with the specified cell set to 0.
    """
    array[mask] = 0
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

def flip(array, axis):
    """
    Flip the array along the specified axis.

    Parameters:
    - array: The input array to be flipped.
    - axis: The axis along which to flip (0 for vertical, 1 for horizontal).

    Returns:
    - Flipped NumPy array.
    """
    array = convert_to_array(array)
    array = np.flip(array, axis)
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