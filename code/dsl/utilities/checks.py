# This file contains functions that check the validity of input arguments.
# Functions:
# - check_integer(value, minimum, maximum): Check if the value is an integer within the specified range. Returns True if valid, False otherwise.
# - check_axis(axis): Check if the axis is valid (1 or 2). Returns True if valid, False otherwise.
# - check_color_rank(rank): Check if the color rank is valid. Returns True if valid, False otherwise.
# - check_color(color): Check if the color is valid. Returns True if valid, False otherwise.

def check_integer(value, minimum, maximum):
    """
    Check if the value is an integer within the specified range. Returns True if valid, False otherwise.
    """
    if isinstance(value, int) == False:
        return False
    if value < minimum or value > maximum:
        return False
    return True

def check_axis(axis):
    """
    Check if the axis is valid (1 or 2). Returns True if valid, False otherwise.
    """
    minimum = 0
    maximum = 1
    return check_integer(axis, minimum, maximum)

def check_color_rank(rank):
    """
    Check if the color rank is valid. Returns True if valid, False otherwise.
    """
    minimum = 0
    maximum = 9
    return check_integer(rank, minimum, maximum)

def check_color(color):
    """
    Check if the color is valid. Returns True if valid, False otherwise.
    """
    minimum = 0
    maximum = 9
    return check_integer(color, minimum, maximum)

def check_num_rotations(num_rotations):
    """
    Check if the number of rotations is valid. Returns True if valid, False otherwise.
    """
    minimum = 1
    maximum = 3
    return check_integer(num_rotations, minimum, maximum)