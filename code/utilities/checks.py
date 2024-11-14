# This file contains functions that check the validity of input arguments.
# Functions:
# - check_axis(axis): Check if the axis is valid (1 or 2). Returns True if valid, False otherwise.
# - check_color_rank(rank, num_colors): Check if the color rank is valid. Returns True if valid, False otherwise.

def check_axis(axis):
    """
    Check if the axis is valid (1 or 2). Returns True if valid, False otherwise.
    """
    if axis not in [1, 2]:
        return False
    return True

def check_color_rank(rank, num_colors):
    """
    Check if the color rank is valid. Returns True if valid, False otherwise.
    """
    if isinstance(rank, int) == False:
        return False
    if rank < 0 or rank >= num_colors:
        return False
    return True