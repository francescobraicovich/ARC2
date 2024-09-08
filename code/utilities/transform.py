
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

def find_smallest_rectangle(array, mask):

    is_mask, js_mask  = np.where(mask)
    i_min, i_max = min(is_mask), max(is_mask)
    j_min, j_max = min(js_mask), max(js_mask)

    return i_min, j_min, i_max, j_max

def find_smallest_square(array, mask):

    i_min, j_min, i_max, j_max = find_smallest_rectangle(array, mask)
    size = max(i_max - i_min + 1, j_max - j_min + 1)

    return i_min, j_min, i_min + size - 1, j_min + size - 1

def rotate(array, mask, n, display=False):
    """
    Rotate the input array 90 degrees n times counterclockwise.

    Parameters:
    - array: The input array to be rotated.
    - n: Number of 90-degree rotations to perform.

    Returns:
    - Rotated NumPy array.
    """

    i_min, j_min, i_max, j_max = find_smallest_square(array, mask)

    array[i_min:i_max+1, j_min:j_max+1] = np.rot90(array[i_min:i_max+1, j_min:j_max+1], n)

    if display:
        plot_array(array)
    
    return array

def delete(array, mask, display=False):
    """
    Set the value of a cell at the given position to 0.

    Parameters:
    - array: The input array.
    - pos: The position of the cell to be deleted.

    Returns:
    - Modified NumPy array with the specified cell set to 0.
    """
    array[mask] = 0
    if display:
        plot_array(array)
    return array

def drag(array, mask, rel_pos_i, rel_pos_j, display=True):
    
    i_min, j_min, i_max, j_max = find_smallest_rectangle(array, mask)
    i_min_new, j_min_new, i_max_new, j_max_new = i_min + rel_pos_i, j_min + rel_pos_j, i_max + rel_pos_i, j_max + rel_pos_j

    if i_min_new < 0 or j_min_new < 0 or i_max_new >= np.shape(array)[0] or j_max_new >= np.shape(array)[1]:
        return array
    
    indices = np.where(mask)
    is_mask, js_mask = indices

    new_is_mask = is_mask + rel_pos_i
    new_js_mask = js_mask + rel_pos_j

    copy = np.copy(array)
    copy[indices] = 0
    copy[new_is_mask, new_js_mask] = array[is_mask, js_mask]

    if display:
        plot_array(copy)

    return copy

def alt_drag(array, mask, rel_pos_i, rel_pos_j, display=True):

    i_min, j_min, i_max, j_max = find_smallest_rectangle(array, mask)
    i_min_new, j_min_new, i_max_new, j_max_new = i_min + rel_pos_i, j_min + rel_pos_j, i_max + rel_pos_i, j_max + rel_pos_j

    if i_min_new < 0 or j_min_new < 0 or i_max_new >= np.shape(array)[0] or j_max_new >= np.shape(array)[1]:
        return array
    
    indices = np.where(mask)
    is_mask, js_mask = indices

    new_is_mask = is_mask + rel_pos_i
    new_js_mask = js_mask + rel_pos_j

    array[new_is_mask, new_js_mask] = array[is_mask, js_mask]

    if display:
        plot_array(array)

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