import numpy as np
import matplotlib.pyplot as plt
from dsl.utilities.checks import check_color, check_integer
from skimage.segmentation import find_boundaries
from scipy.ndimage import label, convolve

# This class is used to select elements of the grid based on specific criteria.

# Implemented methods:
# 1 - select_color: select elements of the grid with a specific color
# 2 - select_rectangles: select rectangles of a specific color, height and width
# 3 - select_connected_shapes: select connected shapes of a specific color
# 4 - select_connected_shapes_diag: select connected shapes of a specific color with diagonal connectivity
# 5 - select_adjacent_to_color: select elements of the grid that are adjacent to a specific color with a specific number of points of contact
# 6 - select_adjacent_to_color_diag: select elements of the grid that are adjacent to a specific color with a specific number of points of contact with diagonal connectivity
# 7 - select_outer_border: select the outer border of the elements with a specific color
# 8 - select_inner_border: select the inner border of the elements with a specific color
# 9 - select_outer_border_diag: select the outer border of the elements with a specific color with diagonal connectivity
# 10 - select_inner_border_diag: select the inner border of the elements with a specific color with diagonal connectivity
# 11 - select_all_grid: select the entire grid


class Selector():
    def __init__(self):
        self.selection_vocabulary = {} # store the selection vocabulary
        self.minimum_geometry_size = 2 # minimum size of the geometry

    def select_color(self, grid:np.ndarray, color:int):
        if check_color(color) == False:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        mask = grid == color
        n_rows, n_cols = grid.shape
        mask = np.reshape(mask, (-1, n_rows, n_cols))
        return mask
    
    def select_rectangles(self, grid, color, height, width):
        """
        Extract all possible rectangles of a given height and width where all elements are True
        from a 2D boolean mask. The result is a 3D array where each layer corresponds to one rectangle.

        Parameters:
        - boolean_mask: 2D numpy array (boolean)
        - height: int, height of the rectangle
        - width: int, width of the rectangle

        Returns:
        - 3D numpy array where each layer contains a single rectangle of True values.
        """
        rows, cols = grid.shape
        rectangles = []

        if check_integer(height, self.minimum_geometry_size, rows) == False:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        if check_integer(width, self.minimum_geometry_size, cols) == False:
            return np.expand_dims(np.zeros_like(grid), axis=0)
      
        color_mask = self.select_color(grid, color)
        
        # if there are no elements with the target color, we return the color mask (all false)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        color_mask = color_mask[0, :, :] # remove the first dimension

        # Iterate over all possible starting points for the rectangle
        for i in range(rows - height + 1):
            for j in range(cols - width + 1):
                # Extract the rectangle
                sub_rect = color_mask[i:i+height, j:j+width]
                
                # Check if all values in the rectangle are True
                if np.all(sub_rect):
                    # Create a new boolean mask with the rectangle as True
                    rect_mask = np.zeros_like(color_mask, dtype=bool)
                    rect_mask[i:i+height, j:j+width] = True
                    rectangles.append(rect_mask)
        
        # Combine all rectangles into a 3D array
        if rectangles:
            result_3d = np.stack(rectangles, axis=0)
        else:
            # If no rectangles are found, return an empty array
            result_3d = np.zeros((0, * color_mask.shape), dtype=bool)
        
        return result_3d
    
    def select_connected_shapes(self, grid, color):
        color_mask = self.select_color(grid, color)
        if np.sum(color_mask) == 0:
            return np.expand_dims(np.zeros_like(grid), axis=0)
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
            return np.expand_dims(np.zeros_like(grid), axis=0)
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
    
    def select_adjacent_to_color(self, grid, color, points_of_contact):
        """
        Finds all cells in a 2D boolean array that have exactly `n` points of contact with `True` values.
        """
        if check_integer(points_of_contact, 1, 4) == False:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        
        nrows, ncols = grid.shape
        
        if nrows == 0 or ncols == 0:
            return np.zeros((0, 0), dtype=bool)
    
        color_mask = self.select_color(grid, color)
        color_mask = color_mask[0, :, :] # remove the first dimension
    
        # Define the kernel for counting neighbors
        kernel = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        
        # Convolve the boolean array with the kernel to count neighbors
        contact_count = convolve(color_mask.astype(int), kernel, mode='constant', cval=0)
        
        # Return a boolean array where contact count equals n
        selection_mask = contact_count == points_of_contact
        selection_mask = selection_mask & ~color_mask
        selection_mask = np.reshape(selection_mask, (-1, nrows, ncols))
        return selection_mask
    
    def select_adjacent_to_color_diag(self, grid, color, points_of_contact):
        """
        Finds all cells in a 2D boolean array that have exactly `n` points of contact with `True` values.
        """
        if check_integer(points_of_contact, 1, 8) == False:
            return np.expand_dims(np.zeros_like(grid), axis=0)
        
        nrows, ncols = grid.shape

        if nrows == 0 or ncols == 0:
            return np.zeros((0, 0), dtype=bool)
        
        color_mask = self.select_color(grid, color)
        color_mask = color_mask[0, :, :] # remove the first dimension
    
        # Define the kernel for counting neighbors
        kernel = np.ones((3, 3), dtype=bool)
        
        # Convolve the boolean array with the kernel to count neighbors
        contact_count = convolve(color_mask.astype(int), kernel, mode='constant', cval=0)
        
        # Return a boolean array where contact count equals n
        selection_mask = contact_count == points_of_contact
        selection_mask = selection_mask & ~color_mask
        selection_mask = np.reshape(selection_mask, (-1, nrows, ncols))
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
    
    def select_all_grid(self, grid, color = None):
        """
        Select the entire grid.
        """
        nrows, ncols = grid.shape
        return np.ones((1, nrows, ncols), dtype=bool)