import numpy as np
import matplotlib.pyplot as plt

def is_valid_geometry(color_mask, start_row, start_col, end_row, end_col):
    """Check if the specified geometry is valid and fully colored."""
    if end_row > color_mask.shape[0] or end_col > color_mask.shape[1]:
        return False
    return np.all(color_mask[start_row:end_row, start_col:end_col])

def find_matching_geometries(color_mask, height, width):
    """Find all geometries matching the specified dimensions in the color mask."""
    matching_geometries = []
    rows, cols = np.where(color_mask)
    
    for start_row, start_col in zip(rows, cols):
        end_row, end_col = start_row + height, start_col + width
        
        if is_valid_geometry(color_mask, start_row, start_col, end_row, end_col):
            matching_geometries.append((start_row, start_col, end_row, end_col))

    return matching_geometries

def geometries_overlap(geo1, geo2):
    """Check if two geometries overlap."""
    r1, c1, r1_end, c1_end = geo1
    r2, c2, r2_end, c2_end = geo2
    
    row_overlap = (r1 <= r2 < r1_end) or (r2 <= r1 < r2_end)
    col_overlap = (c1 <= c2 < c1_end) or (c2 <= c1 < c2_end)
    
    return row_overlap and col_overlap

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

class Selector():
    def __init__(self, shape:tuple, display=False):

        self.shape = shape
        self.nrows, self.ncols = shape
        
        self.selection_vocabulary = {}

        self.display = display
        self.cmap = 'inferno'
        

    def select_color(self, problem:np.ndarray, color:int, display_overturn=None):
        if not isinstance(problem, np.ndarray):
            raise Exception('First argument of the function should be the problem array')
        if (not isinstance(color, int)) or (color < 0) or (color > 9):
            raise Exception('Second argument of the function shoud be the color int, with ')
        mask = problem == color
        mask = np.reshape(mask, (-1, self.nrows, self.ncols))
        
        display = self.display
        if display_overturn != None:
            display = display_overturn
        if display:
            self.plot_selection(mask)
        return mask
    
    def select_colored_rectangle_combinations(self, problem:np.ndarray, color:int, height, width):
        """
        Select elements of the array with a specific color and geometry. Works for rectangular geometries.
        Returns all possible non-overlapping combinations of matching geometries.
        """
        # TODO: Currently this function returns all possible non-overlapping combinations of matching geometris
        # This is computationally too expensive when width and height are small. We need to think of how to deal 
        # with small heights and widths. Implement the solution as checks for height and width values.

        if not isinstance(problem, np.ndarray):
            raise Exception('First argument of the function should be the problem array')
        if (not isinstance(color, int)) or (color < 0) or (color > 9):
            raise Exception('Second argument of the function shoud be the color int, with ')
        
        if height < 1 or width < 1 or height > self.nrows or width > self.ncols:
            mask = np.zeros((1, self.nrows, self.ncols), dtype=bool)
            return mask
      
        color_mask = self.select_color(problem, color, display_overturn=False)
        color_mask = color_mask[0, :, :] # remove the first dimension
        
        # if there are no elements with the target color, we return the color mask (all false)
        if np.sum(color_mask) == 0:
            return color_mask
        
        matching_geometries = find_matching_geometries(color_mask, height, width)

        if len(matching_geometries) == 0:
            mask = np.zeros((1, self.nrows, self.ncols), dtype=bool)
            return mask
        
        geometry_combinations = find_non_overlapping_combinations(matching_geometries)
        
        num_combinations = len(geometry_combinations)
        selection_array = np.zeros((num_combinations, self.nrows, self.ncols), dtype=bool)
        for k, combination in enumerate(geometry_combinations):
            for index in combination:
                i1, j1, i2, j2 = matching_geometries[index]
                selection_array[k, i1:i2, j1:j2] = True

        if self.display:
            self.plot_selection(selection_array)

        return selection_array
    
    def select_colored_separated_shapes(self, problem, color):

        """
        This function selects all shapes of the same color that are not connected one to the other.
        Output: a list of arrays (masks) with the selected geometries.
        """
        color_mask = self.select_color(problem, color, display_overturn=False)
        color_mask = color_mask[0, :, :] # remove the first dimension
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
        selection_array = np.zeros((len(separated_geometries), self.nrows, self.ncols), dtype=int)

        for k, geometry in enumerate(separated_geometries):
            combination_mask = np.zeros(self.shape, dtype=bool)
            for index in geometry:
                i, j = index
                i, j = tuple(index)
                combination_mask[i, j] = True
            selection_array[k] = combination_mask

        if self.display:
            self.plot_selection(selection_array)
        return selection_array
    
    def select_adjacent_to_color(self, problem, color, num_adjacent_cells):

        """
        This function selects cells that are adjacent to a specific color wiht a specific number of points of contact.
        """

        if num_adjacent_cells < 0 or num_adjacent_cells > 4:
            false_mask = np.zeros_like(problem, dtype=bool)
            return false_mask

        color_mask = self.select_color(problem, color, display_overturn=False)
        color_mask = color_mask[0, :, :] # remove the first dimension
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
        
        # add the additional dimension to the selection mask
        selection_mask = np.reshape(convoluted_mask, (-1, self.nrows, self.ncols))

        if self.display:
            self.plot_selection(selection_mask)

        return selection_mask

    def plot_selection(self, selection_mask):
        
        num_selections = len(selection_mask)

        # Calculate the number of rows and columns for the subplots
        num_cols = min(5, num_selections)  # Max 3 columns
        num_rows = (num_selections - 1) // num_cols + 1

        ig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        axs = axs.flatten() if num_selections > 1 else [axs]

        for idx, selection in enumerate(selection_mask):
            axs[idx].imshow(selection, cmap=self.cmap)
            axs[idx].set_title(f'Geometry {idx}')
            axs[idx].axis('off')
    
        # Hide any unused subplots
        for idx in range(num_selections, len(axs)):
            axs[idx].axis('off')

        plt.show()
