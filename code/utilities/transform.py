
import numpy as np

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
    return np.array(array, dtype=int)

def convert_position(array, pos):
    num_cells = np.prod(np.shape(array))
    pos = int(pos % num_cells)
    return pos

def convert_axis(array, axis):
    axis = int(axis % 1)
    return axis

def transpose(array):
    array = convert_to_array(array)
    return np.transpose(array)

def rotate(array, n):
    array = convert_to_array(array)
    return np.rot90(array, n)

def delete_cell(array, pos):
    array = convert_to_array(array)
    pos = convert_position(array, pos)
    i, j = pos // np.shape(array)[1], pos % np.shape(array)[1]
    array[i, j] = 0
    return array

def drag(array, pos1, pos2):
    array = convert_to_array(array)
    pos1, pos2 = convert_position(array, pos1), convert_position(array, pos2)
    i1, j1 = pos1 // np.shape(array)[1], pos1 % np.shape(array)[1]
    i2, j2 = pos2 // np.shape(array)[1], pos2 % np.shape(array)[1]

    array[i2, j2] = array[i1, j1]
    array[i1, j1] = 0
    return array

def alt_drag(array, pos1, pos2):
    array = convert_to_array(array)
    i1, j1 = pos1 // np.shape(array)[1], pos1 % np.shape(array)[1]
    i2, j2 = pos2 // np.shape(array)[1], pos2 % np.shape(array)[1]

    array[i2, j2] = array[i1, j1]
    return array

def flip(array, axis):
    print('original array:', array)
    array = convert_to_array(array)
    array = np.flip(array, axis)
    print('flipped array:', array)
    return array