import numpy as np
import matplotlib.pyplot as plt
from dsl.utilities.checks import check_color, check_integer
from dsl.utilities.selection_utilities import find_matching_geometries, find_non_overlapping_combinations

# This class is used to select elements of the grid based on specific criteria.

# Implemented methods:
# - select_color: select elements of the grid with a specific color
# - select_colored_rectangle_combinations: select elements of the grid with a specific color and geometry
# - select_colored_separated_shapes: select elements of the grid with a specific color that are not connected
# - select_adjacent_to_color: select elements of the grid that are adjacent to a specific color


class Selector():
    def __init__(self, shape:tuple):
        self.shape = shape # shape of the grid
        self.nrows, self.ncols = shape # number of rows and columns
        self.selection_vocabulary = {} # store the selection vocabulary
        self.no_selection = np.zeros((1, self.nrows, self.ncols), dtype=bool) # no selection mask
        self.minimum_geometry_size = 2 # minimum size of the geometry

    def select_color(self, grid:np.ndarray, color:int):
        if check_color(color) == False:
            return self.no_selection
        mask = grid == color
        mask = np.reshape(mask, (-1, self.nrows, self.ncols))
        return mask
    
    def select_colored_rectangle_combinations(self, grid:np.ndarray, color:int, height, width):
        """
        Select elements of the array with a specific color and geometry. Works for rectangular geometries.
        Returns all possible non-overlapping combinations of matching geometries.
        """
        # TODO: Currently this function returns all possible non-overlapping combinations of matching geometris
        # This is computationally too expensive when width and height are small. We need to think of how to deal 
        # with small heights and widths. Implement the solution as checks for height and width values.
        
        if check_integer(height, self.minimum_geometry_size, self.nrows) == False:
            return self.no_selection
        if check_integer(width, self.minimum_geometry_size, self.ncols) == False:
            return self.no_selection
      
        color_mask = self.select_color(grid, color)
        
        # if there are no elements with the target color, we return the color mask (all false)
        if np.sum(color_mask) == 0:
            return self.no_selection
        color_mask = color_mask[0, :, :] # remove the first dimension
        
        matching_geometries = find_matching_geometries(color_mask, height, width)

        if len(matching_geometries) == 0:
            return self.no_selection
        
        geometry_combinations = find_non_overlapping_combinations(matching_geometries)
        
        num_combinations = len(geometry_combinations)
        selection_array = np.zeros((num_combinations, self.nrows, self.ncols), dtype=bool)
        for k, combination in enumerate(geometry_combinations):
            for index in combination:
                i1, j1, i2, j2 = matching_geometries[index]
                selection_array[k, i1:i2, j1:j2] = True
        return selection_array
    
    def select_colored_separated_shapes(self, grid, color):
        pass
    
    def select_colored_separated_shapes(self, grid, color):

        """
        This function selects all shapes of the same color that are not connected one to the other.
        Output: a list of arrays (masks) with the selected geometries.
        """
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return self.no_selection
        color_mask = color_mask[0, :, :] # remove the first dimension
        is_where_true, js_where_true = np.where(color_mask) # get the cordinates of the elements

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
        selection_array = np.zeros((len(separated_geometries), self.nrows, self.ncols), dtype=bool)

        for k, geometry in enumerate(separated_geometries):
            combination_mask = np.zeros(self.shape, dtype=bool)
            for index in geometry:
                i, j = index
                i, j = tuple(index)
                combination_mask[i, j] = True
            selection_array[k] = combination_mask
        return selection_array
    
    def select_adjacent_to_color(self, grid, color, num_adjacent_cells):
        """
        This function selects cells that are adjacent to a specific color wiht a specific number of points of contact.
        """

        if check_integer(num_adjacent_cells, 1, 4) == False:
            return self.no_selection
        
        color_mask = self.select_color(grid, color)
        color_mask = color_mask[0, :, :] # remove the first dimension
        inverse_color_mask = ~color_mask

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
        convoluted_mask = convoluted_mask & inverse_color_mask
        
        # add the additional dimension to the selection mask
        selection_mask = np.reshape(convoluted_mask, (-1, self.nrows, self.ncols))
        return selection_mask
