import numpy as np
from dsl.utilities.checks import check_axis, check_num_rotations
from dsl.utilities.plot import plot_selection, plot_grid
from dsl.utilities.transformation_utilities import create_grid3d, find_bounding_rectangle, find_bounding_square
from dsl.select import Selector


# Transformer class that contains methods to transform the grid.
# The grid is a 2D numpy array, and the selection is a 3D boolean mask.
# Hence, the grid must be stacked along the third dimension to create a 3D grid, using the create_grid3d method.

# Implemented methods:AAA
# - flip(grid, selection, axis): Flip the grid along the specified axis.
# - delete(grid, selection): Set the value of the selected cells to 0.
# - rotate(grid, selection, num_rotations): Rotate the selected cells 90 degrees n times counterclockwise.
# - crop(grid, selection): Crop the grid to the bounding rectangle around the selection. Use -1 as the value for cells outside the selection.
# - fill_with_color(grid, color, fill_color): Fills any shape of a given color with the fill_color

class Transformer:
    def __init__(self):
        pass

    def flip(self, grid, selection, axis):
        """
        Flip the grid along the specified axis.
        """
        grid_3d = create_grid3d(grid, selection) # Add an additional dimension to the grid by stacking it
        if check_axis(axis) == False:
            return grid_3d
        axis += 1 # Increas axis by 1 to account for the additional dimension
        bounding_rectangle = find_bounding_rectangle(selection) # Find the bounding rectangle around the selection for each slice
        flipped_bounding_rectangle = np.flip(bounding_rectangle, axis=axis) # Flip the selection along the specified axis
        grid_3d[bounding_rectangle] = np.flip(grid_3d, axis=axis)[flipped_bounding_rectangle] # Flip the bounding rectangle along the specified axis
        return grid_3d
    
    def delete(self, grid, selection):
        """
        Set the value of the selected cells to 0.
        """
        grid_3d = create_grid3d(grid, selection)
        grid_3d[selection] = 0
        return grid_3d
    
    def rotate(self, grid, selection, num_rotations):
        """
        Rotate the selected cells 90 degrees n times counterclockwise.
        """
        grid_3d = create_grid3d(grid, selection)
        if check_num_rotations(num_rotations) == False:
            return grid_3d
        bounding_square = find_bounding_square(selection)
        rotated_bounding_square = np.rot90(bounding_square, num_rotations, axes=(1, 2))
        grid_3d[bounding_square] = np.rot90(grid_3d, num_rotations, axes=(1, 2))[rotated_bounding_square]
        return grid_3d
    
    def crop(self, grid, selection):
        """
        Crop the grid to the bounding rectangle around the selection. Use -1 as the value for cells outside the selection.
        -1 will be the same number that will be used to pad the grids in order to make them the same size.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)
        grid_3d[~bounding_rectangle] = -1
        return grid_3d
    
    def color(self, grid, selection, color_selected):
        """
        Apply a color transformation (color_selected) to the selected cells (selection) in the grid and return a new 3D grid.
        """
        grid_3d = create_grid3d(grid, selection)

        for idx, mask in enumerate(selection):
            grid_layer = grid.copy()
            grid_layer[mask == 1] = color_selected
            grid_3d[idx] = grid_layer

        return grid_3d

    
    def fill_with_color(self, grid, color, fill_color): #change to take a selection and not do it alone if we want to + 3d or 2d ?
        '''
        Fill all holes inside the single connected shape of the specified color
        and return the modified 2D grid.
        '''
        selector = Selector(grid.shape)
        bounding_shapes = selector.select_colored_separated_shapes(grid, color)  # Get all separated shapes

        # Combine all bounding shapes into one mask
        combined_mask = np.any(bounding_shapes, axis=0)

        # Detect holes inside the combined mask
        from scipy.ndimage import binary_fill_holes
        filled_mask = binary_fill_holes(combined_mask)  # Fill all enclosed regions within the combined mask

        # Create a new grid with the filled mask
        filled_grid = grid.copy()
        filled_grid[filled_mask] = fill_color  # Fill the entire bounding shape (and its holes)

        # Ensure original color (`3`) remains in the bounding region
        filled_grid[combined_mask] = color

        return filled_grid
    

