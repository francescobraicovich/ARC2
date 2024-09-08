import numpy as np
import matplotlib.pyplot as plt

# Plotting functions
def plot_geometries(array, geometries):
    """
    Plot the geometries on the array in different subplots.
    """
    original_array = array.copy()
    num_combinations = len(geometries)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_combinations)  # Max 3 columns
    num_rows = (num_combinations - 1) // num_cols + 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axs = axs.flatten() if num_combinations > 1 else [axs]
    
    for idx, geometry in enumerate(geometries):
        i1, j1, i2, j2 = geometry
        new_array = original_array.copy()
        new_array[i1:i2, j1:j2] = 2
        
        axs[idx].imshow(new_array, cmap='inferno')
        axs[idx].set_title(f'Geometry {idx}')
        axs[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(num_combinations, len(axs)):
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

# Selection functions
def select_by_color(array, color):
    # mask elements of the array
    mask = array == color
    return mask

def select_by_color_and_geometry(array, target_color, height, width, display=False):
    """
    Select elements of the array with a specific color and geometry. Works for rectangular geometries.
    Returns all possible non-overlapping combinations of matching geometries.
    """

    if height < 1 or width < 1:
        return np.zeros_like(array, dtype=bool)
    
    if height > array.shape[0] or width > array.shape[1]:
        return np.zeros_like(array, dtype=bool)

    color_mask = select_by_color(array, target_color)
    
    # if there are no elements with the target color, we return the color mask (all false)
    if np.sum(color_mask) == 0:
        return color_mask
    
    matching_geometries = find_matching_geometries(color_mask, height, width)

    if len(matching_geometries) == 0:
        return np.zeros_like(color_mask, dtype=bool)
    
    if display:
        plot_geometries(array, matching_geometries)

    geometry_combinations = find_non_overlapping_combinations(matching_geometries)

    if display:
        plot_geometry_combinations(array, geometry_combinations, matching_geometries)
    
    num_combinations = len(geometry_combinations)
    geometries_array = np.zeros((num_combinations, array.shape[0], array.shape[1]), dtype=bool)
    for k, combination in enumerate(geometry_combinations):
        for index in combination:
            i1, j1, i2, j2 = matching_geometries[index]
            geometries_array[k, i1:i2, j1:j2] = True

    return geometries_array

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
    
    def recursive_combinations_finder(free_array, current_combinations):
        
        for combination in current_combinations:
            combination_list = list(combination)
            free_row = np.prod(free_array[combination_list], axis=0)
            free_row[combination_list] = False
            
            if any(free_row):
                indices_where_true = np.where(free_row)[0]
                new_combinations = [combination.copy() for _ in range(len(indices_where_true))]
                for i, index in enumerate(indices_where_true):
                    new_combinations[i].add(int(index))
  
                new_combinations = recursive_combinations_finder(free_array, new_combinations)
                current_combinations.extend(new_combinations)
        
        return current_combinations

    combinations = recursive_combinations_finder(free_array, [set([i]) for i in range(len(geometries))])
    combinations = set([frozenset(combination) for combination in combinations])

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

""""
def select_by_color_and_area(array, target_color, area):

    color_mask = select_by_color(array, target_color)
    geometries = find_matching_geometries(color_mask, 1, 1)
    num_geometries = len(geometries)

    indices_where_true = np.where(color_mask)
    selected_geometries = []
    
    for geometry in geometries:
        i1, j1, i2, j2 = geometry
        if (i2 - i1) * (j2 - j1) == area:
            selected_geometries.append(geometry)
    
    return selected_geometries
"""

def select_by_color_and_adjacent(array, target_color, display=False):
    """
    This function selects geometries of the same color that are adjacent to each other.
    Output: a list of arrays (masks) with the selected geometries.
    """
    color_mask = select_by_color(array, target_color)
    is_where_true, js_where_true = np.where(color_mask)

    if np.sum(color_mask) == 0:
        return np.zeros_like(color_mask, dtype=bool)

    separated_geometries = [] # store the separated geometries
    separated_geometries.append({(is_where_true[0], js_where_true[0])}) # add the first element
    
    for i, j in zip(is_where_true[1:], js_where_true[1:]): # iterate over the rest of the elements
        found = False # flag to check if the element is found in the separated geometries
        for geometry in separated_geometries: # iterate over the separated geometries
            for tuple_cordinates in geometry: # iterate over the cordinates of the geometry
                i0, j0 = tuple_cordinates # get the cordinates of the element in the geometry
                if abs(i-i0) <= 1 and abs(j-j0) <= 1: 
                    geometry.add((i, j))
                    found = True
                    break
                
        if not found:
            separated_geometries.append({(i, j)})

    # create an array to store the selected geometries
    selected_geometries = np.zeros((len(separated_geometries), array.shape[0], array.shape[1]), dtype=int)

    for k, geometry in enumerate(separated_geometries):
        combination_mask = np.zeros(array.shape, dtype=bool)
        for index in geometry:
            i, j = index
            i, j = tuple(index)
            combination_mask[i, j] = True
        if display:
            plt.imshow(combination_mask, cmap='inferno')
            plt.title(f'Geometry {k}')
            plt.show()
        selected_geometries[k] = combination_mask

    return selected_geometries

def select_adjacent_to_color(array, target_color, num_adjacent_cells, display=False):

    """
    This function selects cells that are adjacent to a specific color wiht a specific number of points of contact.
    """

    if num_adjacent_cells < 0 or num_adjacent_cells > 4:
        false_mask = np.zeros_like(array, dtype=bool)
        return false_mask

    color_mask = select_by_color(array, target_color)
    invers_color_mask = ~color_mask

    # create a padded color mask
    padded_color_mask = np.pad(color_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # convolute the color mask with the kernel
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    convoluted_mask = np.zeros_like(padded_color_mask, dtype=bool)

    for i in range(padded_color_mask.shape[0]-2):
        for j in range(padded_color_mask.shape[1]-2):
            convoluted_mask[i, j] = np.sum(padded_color_mask[i:i+3, j:j+3] * kernel) == num_adjacent_cells

    # remove the padding
    convoluted_mask = convoluted_mask[:-2, :-2]
    convoluted_mask = convoluted_mask & invers_color_mask

    if display:
        plt.imshow(convoluted_mask, cmap='inferno')
        plt.title('Selected geometries')
        plt.show()

    return convoluted_mask
    


    