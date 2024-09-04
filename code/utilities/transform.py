
import numpy as np


def select_by_color(array, color):
    # mask elements of the array
    mask = array == color
    return mask

def select_by_color_and_geometry(array, color, l1, l2):

    """
    This function selects the elements of the array that are of a certain color and have a certain geometry.
    If more than one selection is possible, both are returned. Both will be kept as a possible transformation.
    The utility function will then have to decide which one to keep. 
    """

    # mask elements of the array
    color_mask = select_by_color(array, color)
    indices_color = np.where(color_mask) # indices of the elements of the array that are of the color we are looking for

    all_geometries = []

    # iterate over the indices of the elements of the array that are of the color we are looking for 
    # to find all matching geometries of the color we are looking for. 
    for i, j in zip(*indices_color):
        
        delta_i = l1
        delta_j = l2

        new_i = i + delta_i
        new_j = j + delta_j

        new_i_in_bounds = new_i >= 0 and new_i < array.shape[0]
        new_j_in_bounds = new_j >= 0 and new_j < array.shape[1]

        if new_i_in_bounds and new_j_in_bounds: # if the new indices are in bounds, we can select the geometry
            geometry = color_mask[i:new_i, j:new_j] # select the geometry in the color mask

            if np.sum(geometry) == l1*l2: # if the geometry is all of the color we are looking for, we can return it
                all_geometries.append((i, j, new_i, new_j)) # return the indices of the geometry

    def overlap(geometry1, geometry2):
        """
        Check if two geometries overlap.
        """
        i1, j1, new_i1, new_j1 = geometry1
        i2, j2, new_i2, new_j2 = geometry2

        # check if the geometries overlap
        overlap_i = (i1 >= i2 and i1 <= new_i2) or (new_i1 >= i2 and new_i1 <= new_i2) or (i2 >= i1 and i2 <= new_i1) or (new_i2 >= i1 and new_i2 <= new_i1)
        overlap_j = (j1 >= j2 and j1 <= new_j2) or (new_j1 >= j2 and new_j1 <= new_j2) or (j2 >= j1 and j2 <= new_j1) or (new_j2 >= j1 and new_j2 <= new_j1)

        return overlap_i and overlap_j

    geometries_combinations = set() # store all possible combinations of the geometries that do not overlap
    
    # find all possible combinations of the geometries that do not overlap
    for i, geometry1 in enumerate(all_geometries): # iterate over all geometries
        geometry_combination_i = set(i) # store the indices of the geometries that do not overlap with geometry1
        
        for j, geometry2 in enumerate(all_geometries): # iterate over all geometries again
            overlap_found = False
            for k in geometry_combination_i: # iterate over the indices of the geometries that are already in the combination
                selected_geometry = all_geometries[k] # get the geometry that is already in the combination
                if overlap(geometry2, selected_geometry): # check if the geometry2 overlaps with the selected geometry
                    overlap_found = True # if there is an overlap, we break
                    break
            if not overlap_found: # if there is no overlap, we add the geometry to the combination
                geometry_combination_i.add(j) 
        geometry_combination_i = frozenset(geometry_combination_i) # convert the set to a frozenset to be able to add it to the set of geometries combinations
        geometries_combinations.add(geometry_combination_i) # add the combination to the set of geometries combinations

    geometries_combinations = list(geometries_combinations) # convert the set to a list
    return geometries_combinations







        





            

                    

        











def apply_transformations(array, transformations, kwargs):
    """
    Apply a series of transformations to an array.

    Parameters:
    - array: The input array to be transformed.
    - transformations: A list of transformation functions to be applied to the array.
    - kwargs: A list of keyword arguments for each transformation function. If None, no keyword arguments will be passed.

    Returns:
    - The transformed array.

    """
    for j, transformation in enumerate(transformations):
        if kwargs is not None:
            current_kwargs = kwargs[j] if kwargs is not None else {}
            array = transformation(array, **current_kwargs)
        else:
            array = transformation(array)
    return array

def convert_to_array(array):
    """
    Convert input to a NumPy array of integers.

    Parameters:
    - array: Input data to be converted.

    Returns:
    - NumPy array of integers.
    """
    return np.array(array, dtype=int)

def convert_position(array, pos):
    """
    Convert a position to a valid index within the array.

    Parameters:
    - array: The input array.
    - pos: The position to be converted.

    Returns:
    - An integer representing a valid position within the array.
    """
    num_cells = np.prod(np.shape(array))
    pos = int(pos % num_cells)
    return pos

def convert_axis(array, axis):
    """
    Convert an axis value to either 0 or 1.

    Parameters:
    - array: The input array (unused in this function).
    - axis: The axis value to be converted.

    Returns:
    - An integer (0 or 1) representing the axis.
    """
    axis = int(axis % 2)
    return axis

def convert_color(array, color):
    """
    Convert a color value to a valid color index (0-8).

    Parameters:
    - array: The input array (unused in this function).
    - color: The color value to be converted.

    Returns:
    - An integer (0-8) representing a valid color index.
    """
    color = int(color % 9)
    return color

def transpose(array):
    """
    Transpose the input array.

    Parameters:
    - array: The input array to be transposed.

    Returns:
    - Transposed NumPy array.
    """
    array = convert_to_array(array)
    return np.transpose(array)

def rotate(array, n):
    """
    Rotate the input array 90 degrees n times counterclockwise.

    Parameters:
    - array: The input array to be rotated.
    - n: Number of 90-degree rotations to perform.

    Returns:
    - Rotated NumPy array.
    """
    array = convert_to_array(array)
    return np.rot90(array, n)

def delete_cell(array, pos):
    """
    Set the value of a cell at the given position to 0.

    Parameters:
    - array: The input array.
    - pos: The position of the cell to be deleted.

    Returns:
    - Modified NumPy array with the specified cell set to 0.
    """
    array = convert_to_array(array)
    pos = convert_position(array, pos)
    i, j = pos // np.shape(array)[1], pos % np.shape(array)[1]
    array[i, j] = 0
    return array

def drag(array, pos1, pos2):
    """
    Move a cell from pos1 to pos2, leaving the original position empty.

    Parameters:
    - array: The input array.
    - pos1: The original position of the cell to be moved.
    - pos2: The destination position for the cell.

    Returns:
    - Modified NumPy array with the cell moved.
    """
    array = convert_to_array(array)
    pos1, pos2 = convert_position(array, pos1), convert_position(array, pos2)
    i1, j1 = pos1 // np.shape(array)[1], pos1 % np.shape(array)[1]
    i2, j2 = pos2 // np.shape(array)[1], pos2 % np.shape(array)[1]

    array[i2, j2] = array[i1, j1]
    array[i1, j1] = 0
    return array

def alt_drag(array, pos1, pos2):
    """
    Copy a cell from pos1 to pos2, leaving the original cell unchanged.

    Parameters:
    - array: The input array.
    - pos1: The position of the cell to be copied.
    - pos2: The destination position for the copy.

    Returns:
    - Modified NumPy array with the cell copied.
    """
    array = convert_to_array(array)
    i1, j1 = pos1 // np.shape(array)[1], pos1 % np.shape(array)[1]
    i2, j2 = pos2 // np.shape(array)[1], pos2 % np.shape(array)[1]

    array[i2, j2] = array[i1, j1]
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