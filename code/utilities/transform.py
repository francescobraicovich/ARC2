
import numpy as np
import matplotlib.pyplot as plt

# Selection functions
def select_by_color(array, color):
    # mask elements of the array
    mask = array == color
    return mask

def plot_geometries(array, geometries):
    """
    Plot the geometries on the array in different subplots.
    """
    original_array = array.copy()
    num_geometries = len(geometries)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_geometries)  # Max 3 columns
    num_rows = (num_geometries - 1) // num_cols + 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axs = axs.flatten() if num_geometries > 1 else [axs]
    
    for idx, geometry in enumerate(geometries):
        i1, j1, i2, j2 = geometry
        new_array = original_array.copy()
        new_array[i1:i2, j1:j2] = 2
        
        axs[idx].imshow(new_array, cmap='inferno')
        axs[idx].set_title(f'Geometry {idx}')
        axs[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(num_geometries, len(axs)):
        axs[idx].axis('off')
    
    # add a global title
    plt.suptitle('Geometries matching the color')
    plt.show()

def plot_geometry_combinations(array, geometries_combinations, all_geometries):
    """
    Plot the geometry combinations on the array in different subplots.
    """
    original_array = array.copy()
    num_combinations = len(geometries_combinations)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_combinations)  # Max 5 columns
    num_rows = (num_combinations - 1) // num_cols + 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axs = axs.flatten() if num_combinations > 1 else [axs]
    
    for idx, geometry_combination in enumerate(geometries_combinations):
        new_array = original_array.copy()
        for j in geometry_combination:
            i1, j1, i2, j2 = all_geometries[j]
            new_array[i1:i2, j1:j2] = 2 + j
        
        axs[idx].imshow(new_array, cmap='inferno')
        axs[idx].set_title(f'Combination {idx+1}')
        axs[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(num_combinations, len(axs)):
        axs[idx].axis('off')
    
    plt.suptitle('Geometries combinations matching the color')
    plt.show()


def select_by_color_and_geometry(array, target_color, height, width, display=False):
    """
    Select elements of the array with a specific color and geometry.
    Returns all possible non-overlapping combinations of matching geometries.
    """
    color_mask = select_by_color(array, target_color)
    matching_geometries = find_matching_geometries(color_mask, height, width)
    
    if display:
        plot_geometries(array, matching_geometries)

    geometry_combinations = find_non_overlapping_combinations(matching_geometries)

    if display:
        plot_geometry_combinations(array, geometry_combinations, matching_geometries)

    return geometry_combinations

def find_matching_geometries(color_mask, height, width, display=False):
    """Find all geometries matching the specified dimensions in the color mask."""
    matching_geometries = []
    rows, cols = np.where(color_mask)
    
    for start_row, start_col in zip(rows, cols):
        end_row, end_col = start_row + height, start_col + width
        
        if is_valid_geometry(color_mask, start_row, start_col, end_row, end_col):
            matching_geometries.append((start_row, start_col, end_row, end_col))

    if display:
        plot_geometries(color_mask, matching_geometries)
    
    return matching_geometries

def is_valid_geometry(color_mask, start_row, start_col, end_row, end_col):
    """Check if the specified geometry is valid and fully colored."""
    if end_row > color_mask.shape[0] or end_col > color_mask.shape[1]:
        return False
    return np.all(color_mask[start_row:end_row, start_col:end_col])

def find_non_overlapping_combinations(geometries):
    """Find all possible combinations of non-overlapping geometries."""    

    free_array = np.zeros((len(geometries), len(geometries)), dtype=bool)
    for i in range(len(geometries)):
        for j in range(i+1, len(geometries)):
            free_array[i, j] = not geometries_overlap(geometries[i], geometries[j])
    print('overlap_array: \n', free_array)
    overlap_array = np.logical_not(free_array)
    
    
    
    def recursive_combinations_finder(free_array, current_combinations):
        print(f'current combinations: ', current_combinations)
        
        for combination in current_combinations:
            print(f'combination: ', combination)
            combination_list = list(combination)
            free_row = np.prod(free_array[combination_list], axis=0)
            free_row[combination_list] = False
            
            if any(free_row):
                indices_where_true = np.where(free_row)[0]
                print(f'indices where true: ', indices_where_true)
                new_combinations = [combination.copy() for _ in range(len(indices_where_true))]
                for i, index in enumerate(indices_where_true):
                    new_combinations[i].add(int(index))
                #current_combinations.remove(combination)
                print(f'new combinations: ', new_combinations)
                new_combinations = recursive_combinations_finder(free_array, new_combinations)
                current_combinations.extend(new_combinations)
            
        print(f'current combinations: ', current_combinations)
        print('')
        
        return current_combinations

    combinations = recursive_combinations_finder(free_array, [set([i]) for i in range(len(geometries))])
    combinations = set([frozenset(combination) for combination in combinations])
    print(f'combinations: ', combinations)
    """
    combinations = set()
    for i in range(len(geometries)-1, -1, -1):
        combination = set([i])
        for j in range(len(geometries)-1, -1, -1):
            overlap_found = False
            for k in combination:
                if overlap_array[j, k]:
                    overlap_found = True
                    break
            if not overlap_found:
                combination.add(j)
        combinations.add(frozenset(combination))
    combinations = list(combinations)
    
    for p in range(num_permutations): #iterate over all orders of geometries
        new_order = permutation_array[p]
        for i in new_order:
            combination = find_compatible_geometries(geometries, i)
           
            combinations.add(frozenset(combination))
    """
    return list(combinations)

def find_compatible_geometries(geometries, start_index):
    """Find all geometries compatible with the starting geometry."""
    compatible = {start_index}
    start_geometry = geometries[start_index]

    for j, candidate in enumerate(geometries): # iterate over all geometries again
            overlap_found = False
            for k in compatible: # iterate over the indices of the geometries that are already in the combination
                compatible_k = geometries[k] # get the geometry that is already in the combination
                print(f'checking goemetries {j} and {k}')
                if geometries_overlap(candidate, compatible_k): # check if the geometry2 overlaps with the selected geometry
                    overlap_found = True # if there is an overlap, we break
                    print(f'overlap')
                    break
                else:
                    print(f'free')
                print(f'compatible: {compatible}\n')
            if not overlap_found: # if there is no overlap, we add the geometry to the combination
                compatible.add(j)
    
    return compatible

def geometries_overlap(geo1, geo2):
    """Check if two geometries overlap."""
    r1, c1, r1_end, c1_end = geo1
    r2, c2, r2_end, c2_end = geo2
    
    row_overlap = (r1 <= r2 < r1_end) or (r2 <= r1 < r2_end)
    col_overlap = (c1 <= c2 < c1_end) or (c2 <= c1 < c2_end)
    
    return row_overlap and col_overlap







        





            

                    

        











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