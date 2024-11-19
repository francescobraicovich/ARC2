import numpy as np
import matplotlib.pyplot as plt
from dsl.utilities.checks import check_color, check_integer
from dsl.utilities.selection_utilities import find_matching_geometries, find_non_overlapping_combinations
from skimage.segmentation import find_boundaries
from scipy.ndimage import label

# This class is used to select elements of the grid based on specific criteria.

# Implemented methods:
# - select_color: select elements of the grid with a specific color
# - select_colored_rectangle_combinations: select elements of the grid with a specific color and geometry
# - select_connected_shapes: select connected shapes of a specific color
# - select_connected_shapes_diag: select connected shapes of a specific color with diagonal connectivity
# - select_adjacent_to_color: select elements of the grid that are adjacent to a specific color
# - select_outer_border: select the outer border of the elements with a specific color
# - select_inner_border: select the inner border of the elements with a specific color
# - select_outer_border_diag: select the outer border of the elements with a specific color with diagonal connectivity
# - select_inner_border_diag: select the inner border of the elements with a specific color with diagonal connectivity


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
    
    def select_connected_shapes(self, grid, color):
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return self.no_selection
        color_mask = color_mask[0, :, :] # remove the first dimension

        # Label connected components
        labeled_array, num_features = label(color_mask)
        
        # Initialize the 3D array with the same height and width as the input
        shape = (num_features, *color_mask.shape)
        result_3d = np.zeros(shape, dtype=bool)
        
        # Extract each connected component as a separate 2D array
        for i in range(1, num_features + 1):  # Labels start from 1
            result_3d[i - 1] = (labeled_array == i)
        
        return result_3d
    
    def select_connected_shapes_diag(self, grid, color):
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return self.no_selection
        color_mask = color_mask[0, :, :] # remove the first dimension
        
        # Label connected components
        structure = np.ones((3, 3), dtype=bool)  # 8-connectivity for 2D
        labeled_array, num_features = label(color_mask, structure)
        
        # Initialize the 3D array with the same height and width as the input
        shape = (num_features, *color_mask.shape)
        result_3d = np.zeros(shape, dtype=bool)
        
        # Extract each connected component as a separate 2D array
        for i in range(1, num_features + 1):  # Labels start from 1
            result_3d[i - 1] = (labeled_array == i)
        
        return result_3d
    
    
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

    def select_outer_border(self, grid, color):
        """
        Select the outer border of the elements with a specific color.
        """
        color_separated_shapes = self.select_connected_shapes(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode = 'outer')
        return color_separated_shapes
    
    def select_inner_border(self, grid, color):
        """
        Select the inner border of the elements with a specific color.
        """
        color_separated_shapes = self.select_connected_shapes(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode = 'inner')
        return color_separated_shapes
    
    def select_outer_border_diag(self, grid, color):
        """
        Select the outer border of the elements with a specific color with diagonal connectivity.
        """
        color_separated_shapes = self.select_connected_shapes_diag(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode = 'outer')
        return color_separated_shapes
    
    def select_inner_border_diag(self, grid, color):
        """
        Select the inner border of the elements with a specific color with diagonal connectivity.
        """
        color_separated_shapes = self.select_connected_shapes_diag(grid, color)
        for i in range(len(color_separated_shapes)):
            color_separated_shapes[i] = find_boundaries(color_separated_shapes[i], mode = 'inner')
        return color_separated_shapes