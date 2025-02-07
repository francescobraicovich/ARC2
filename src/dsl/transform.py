import numpy as np
from dsl.utilities.plot import plot_selection
from dsl.utilities.checks import check_axis, check_num_rotations, check_color, check_integer
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from dsl.utilities.transformation_utilities import create_grid3d, find_bounding_rectangle, find_bounding_square, center_of_mass, vectorized_center_of_mass, missing_integer
from dsl.select import Selector
from dsl.color_select import ColorSelector


# Transformer class that contains methods to transform the grid.
# The grid is a 2D numpy array, and the selection is a 3D boolean mask.
# Hence, the grid must be stacked along the third dimension to create a 3D grid, using the create_grid3d method.

# Implemented methods:
# 1 - flipv(grid, selection): Flip the grid vertically.
# 2 - fliph(grid, selection): Flip the grid horizontally.
# 3 - delete(grid, selection): Set the value of the selected cells to 0.
# 4 - rotate90(grid, selection): Rotate the selected cells 90 degrees counterclockwise.
# 5 - rotate180(grid, selection): Rotate the selected cells 180 degrees counterclockwise.
# 6 - rotate270(grid, selection): Rotate the selected cells 270 degrees counterclockwise.
# 7 - crop(grid, selection): Crop the grid to the bounding rectangle around the selection. Use -1 as the value for cells outside the selection.
# 8 - fill_with_color(grid, color, fill_color): Fills any shape of a given color with the fill_color
# 9 - mirror_main_diagonal(grid, selection): Mirror the selected region along the main diagonal (top-left to bottom-right).
# 10 - mirror_anti_diagonal(grid, selection): Mirror the selected region along the anti-diagonal (top-right to bottom-left).
# 11 - color(grid, selection, color_selected): Apply a color transformation (color_selected) to the selected cells (selection) in the grid and return a new 3D grid.   
# 12 - copy_paste(grid, selection, shift_x, shift_y): Shift the selected cells in the grid by (shift_x, shift_y).
# 13 - copy_sum(grid, selection, shift_x, shift_y): Shift the selected cells in the grid by (shift_x, shift_y) without using loops and sum the values.
# 14 - cut_paste(grid, selection, shift_x, shift_y): Shift the selected cells in the grid by (shift_x, shift_y) and set the original cells to 0.
# 15 - cut_sum(grid, selection, shift_x, shift_y): Shift the selected cells in the grid by (shift_x, shift_y) without using loops and sum the values.
# 16 - change_background_color(grid, selection, new_color): Change the background color of the grid to the specified color.
# 17 - change_selection_to_background_color(grid, selection): Change the selected cells in the grid to the background color.
# 18 - vupscale(grid, selection, scale_factor): Upscale the selection in the grid by a specified scale factor, and cap the upscaled selection to match the original size.
# 19 - hupscale(grid, selection, scale_factor): Upscale the selection in the grid by a specified scale factor, and cap the upscaled selection to match the original size.
# 20 - fill_bounding_rectangle_with_color(grid, selection, color): Fill the bounding rectangle around the selection with the specified color.
# 21 - fill_bounding_square_with_color(grid, selection, color): Fill the bounding square around the selection with the specified color.
# 22/23 - mirror_up/down(grid, selection): Mirrors the selection up/down out of the original grid. Works only id columns <= 15.
# 24/25 - mirror_left/right(grid, selection): Mirrors the selection left/right out of the original grid. Works only id rows <= 15.
# 26 - duplicate_horizontally(grid, selection): Duplicate the selection horizontally out of the original grid. Works only if columns <= 15.
# 27 - duplicate_vertically(grid, selection): Duplicate the selection vertically out of the original grid. Works only if rows <= 15. 
# 28 - copy_paste_vertically(grid, selection): For each mask in the selection, copy its selected area and paste it upwards and downwards as many times as possible within the grid bounds.
# 29 - copy_paste_horizontally(grid, selection): For each mask in the selection, copy its selected area and paste it leftwards and rightwards as many times as possible within the grid bounds.
# XX - vectorized_vupscale the selection in the grid by a specified scale factor, and cap the upscaled selection to match the original size. #TODO @francesco please review method and compare it to the vupscale method
# 30/33 - gravitate_whole_direction_paste(grid, selection, direction): Copies and pastes the whole selection in the grid along the specified direction until either the end of the grid or the first object encounteres in that direction.
# 34/37 - gravitate_whole_direction_cut(grid, selection, direction): Cuts and pastes the whole selection in the grid along the specified direction until either the end of the grid or the first object encounteres in that direction.
#38/40 - direction_gravity(grid, selection): Move the selection in the direction until it hits the edge of the grid or another object. Tiles of the selection are moved independently, they are not "glued".



def select_color(grid, method, param):
    colsel = ColorSelector()
    if method == 'color_rank':
        return colsel.rankcolor(grid, param)
    
    if method == 'shape_rank_nodiag':
        return colsel.rank_largest_shape_color_nodiag(grid, param)
    
    if method == 'shape_rank_diag':
        return colsel.rank_largest_shape_color_diag(grid, param)
    
class Transformer:
    def __init__(self):
        pass
    
  # Coloring transformations
    def new_color(self, grid, selection, color):
        """
        Change the color of the selected cells in the grid to the specified color.
        """
        grid_3d = create_grid3d(grid, selection)
        if np.sum(grid == color) == 0:
            grid_3d[selection == 1] = color
            return grid_3d
        else:
            return np.expand_dims(grid, axis=0)
  
    def color(self, grid, selection, method, param):
        """
        Apply a color transformation (color_selected) to the selected cells (selection) in the grid and return a new 3D grid.
        """
        color_selected = select_color(grid, method, param)
        grid_3d = create_grid3d(grid, selection)
        grid_3d[selection == 1] = color_selected
        return grid_3d

    def fill_with_color(self, grid, selection, method, param): #change to take a selection and not do it alone if we want to + 3d or 2d ?
        '''
        Fill all holes inside the single connected shape of the specified color
        and return the modified 2D grid.
        '''
        grid_3d = create_grid3d(grid, selection)  
        fill_color = select_color(grid, method, param)
        if check_color(fill_color) == False:
            return grid_3d
        filled_masks = np.array([binary_fill_holes(i) for i in selection])
        # Fill the holes in the grids with the specified color
        new_masks = filled_masks & (~selection)
        grid_3d[new_masks] = fill_color

        return grid_3d
    
    def fill_bounding_rectangle_with_color(self, grid, selection, method, param):
        '''
        Fill the bounding rectangle around the selection with the specified color.
        '''
        color = select_color(grid, method, param)
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)
        grid_3d = np.where((bounding_rectangle & (bounding_rectangle & (1-selection))) == 1, color, grid_3d)
        return grid_3d
    
    def fill_bounding_square_with_color(self, grid, selection, method, param):
        '''
        Fill the bounding square around the selection with the specified color.
        '''
        color = select_color(grid, method, param)
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)
        grid_3d = np.where((bounding_square & (bounding_square & (1-selection))) == 1, color, grid_3d)
        return grid_3d

    # Flipping transformations
    def flipv(self, grid, selection):
        """
        Flip the grid along the specified axis.
        """
        grid_3d = create_grid3d(grid, selection) # Add an additional dimension to the grid by stacking it
        bounding_rectangle = find_bounding_rectangle(selection) # Find the bounding rectangle around the selection for each slice
        flipped_bounding_rectangle = np.flip(bounding_rectangle, axis=1) # Flip the selection along the specified axis
        grid_3d[bounding_rectangle] = np.flip(grid_3d, axis=1)[flipped_bounding_rectangle] # Flip the bounding rectangle along the specified axis
        return grid_3d
    
    def fliph(self, grid, selection):
        """
        Flip the grid along the specified axis.
        """
        grid_3d = create_grid3d(grid, selection) # Add an additional dimension to the grid by stacking it
        bounding_rectangle = find_bounding_rectangle(selection) # Find the bounding rectangle around the selection for each slice
        flipped_bounding_rectangle = np.flip(bounding_rectangle, axis=2) # Flip the selection along the specified axis
        grid_3d[bounding_rectangle] = np.flip(grid_3d, axis=2)[flipped_bounding_rectangle] # Flip the bounding rectangle along the specified axis
        return grid_3d
    
    def flip_main_diagonal(self, grid, selection):
        '''
        Mirror the selected region along the main diagonal (top-left to bottom-right).
        '''
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)  # Find the bounding square for each selection slice

        for i in range(grid_3d.shape[0]):  # Iterate through each selection slice
            mask = bounding_square[i]  # Mask for the current bounding square
            rows, cols = np.where(mask)  # Get the indices of the selected region
            if len(rows) > 0 and len(cols) > 0 and rows.max() - rows.min() == cols.max() - cols.min():
                # Calculate the bounding square limits
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()

                # Extract the square region
                square = grid_3d[i, min_row:max_row+1, min_col:max_col+1]
                # Mirror along the main diagonal
                mirrored = square.T
                # Replace the original square with the mirrored one
                grid_3d[i, min_row:max_row+1, min_col:max_col+1] = mirrored

        return grid_3d

    def flip_anti_diagonal(self, grid, selection):
        '''
        Mirror the selected region along the anti-diagonal (top-right to bottom-left).
        '''
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)  # Find the bounding square for each selection slice

        for i in range(grid_3d.shape[0]):  # Iterate through each selection slice
            mask = bounding_square[i]  # Mask for the current bounding square
            rows, cols = np.where(mask)  # Get the indices of the selected region
            if len(rows) > 0 and len(cols) > 0 and rows.max() - rows.min() == cols.max() - cols.min():
                # Calculate the bounding square limits
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()

                # Extract the square region
                square = grid_3d[i, min_row:max_row+1, min_col:max_col+1].copy()
                # Mirror along the anti-diagonal
                mirrored = np.flip((np.rot90(square)),1)
                # Replace the original square with the mirrored one
                grid_3d[i, min_row:max_row+1, min_col:max_col+1] = mirrored

        return grid_3d


    # Rotation transformations
    def rotate(self, grid, selection, num_rotations):
        """
        Rotate the selected cells 90 degrees num_rotations times counterclockwise.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_masks = find_bounding_square(selection)

        for i in range(bounding_masks.shape[0]):
            # Get the bounding square mask for the current layer
            bounding_mask = bounding_masks[i]

            # Identify rows and columns of the bounding square
            rows, cols = np.where(bounding_mask)
            if rows.size == 0 or cols.size == 0:
                continue  # Skip empty bounding squares
            if rows.max() - rows.min() == cols.max() - cols.min():
                
                row_start, row_end = rows.min(), rows.max() + 1
                col_start, col_end = cols.min(), cols.max() + 1

                # Extract the sub-grid corresponding to the bounding square
                sub_grid = grid_3d[i, row_start:row_end, col_start:col_end]

                # Rotate the sub-grid
                rotated_sub_grid = np.rot90(sub_grid, num_rotations)

                # Place the rotated sub-grid back into the grid
                grid_3d[i, row_start:row_end, col_start:col_end] = rotated_sub_grid

            else:
                continue # Skip non-square bounding regions

        return grid_3d
    
    def rotate_90(self, grid, selection):
        """
        Rotate the selected cells 90 degrees counterclockwise.
        """
        return self.rotate(grid, selection, 1)

    def rotate_180(self, grid, selection):
        """
        Rotate the selected cells 180 degrees counterclockwise.
        """
        return self.rotate(grid, selection, 2)
    
    def rotate_270(self, grid, selection):
        """
        Rotate the selected cells 270 degrees counterclockwise.
        """
        return self.rotate(grid, selection, 3)
    
    
    # Mirror and duplicate transformations
    def mirror_down(self, grid, selection):
        """
        Mirrors the selection vertically below the original grid.
        Works only if rows <= 15. If rows > 15, returns the grid in 3D form.
        """
        d, rows, cols = np.shape(selection)
        grid_3d = create_grid3d(grid, selection)

        if rows > 15:
            return grid_3d

        new_grid_3d = np.zeros((d, rows * 2, cols))
        new_grid_3d[:, :rows, :] = grid_3d
        new_grid_3d[:, rows:, :] = np.flip(grid_3d, axis=1)
        flipped_selection = np.flip(selection, axis=1).astype(bool)
        new_grid_3d[:, rows:, :][~flipped_selection] = 0
        return new_grid_3d.astype(int)

    def mirror_up(self, grid, selection):
        """
        Mirrors the selection vertically above the original grid.
        Works only if rows <= 15. If rows > 15, returns the grid in 3D form.
        """
        d, rows, cols = np.shape(selection)
        grid_3d = create_grid3d(grid, selection)

        if rows > 15:
            return grid_3d

        new_grid_3d = np.zeros((d, rows * 2, cols))
        new_grid_3d[:, :rows, :] = np.flip(grid_3d, axis=1)
        new_grid_3d[:, rows:, :] = grid_3d
        flipped_selection = np.flip(selection, axis=1).astype(bool)
        new_grid_3d[:, :rows, :][~flipped_selection] = 0
        return new_grid_3d.astype(int)

    def mirror_right(self, grid, selection):
        """
        Mirrors the selection horizontally to the right of the original grid.
        Works only if columns <= 15. If columns > 15, returns the grid in 3D form.
        """
        d, rows, cols = np.shape(selection)
        grid_3d = create_grid3d(grid, selection)

        if cols > 15:
            return grid_3d

        new_grid_3d = np.zeros((d, rows, cols * 2))
        new_grid_3d[:, :, :cols] = grid_3d
        new_grid_3d[:, :, cols:] = np.flip(grid_3d, axis=2)
        flipped_selection = np.flip(selection, axis=2).astype(bool)
        new_grid_3d[:, :, cols:][~flipped_selection] = 0
        return new_grid_3d.astype(int)

    def mirror_left(self, grid, selection):
        """
        Mirrors the selection horizontally to the left of the original grid.
        Works only if columns <= 15. If columns > 15, returns the grid in 3D form.
        """
        d, rows, cols = np.shape(selection)
        grid_3d = create_grid3d(grid, selection)

        if cols > 15:
            return grid_3d

        new_grid_3d = np.zeros((d, rows, cols * 2))
        new_grid_3d[:, :, :cols] = np.flip(grid_3d, axis=2)
        new_grid_3d[:, :, cols:] = grid_3d
        flipped_selection = np.flip(selection, axis=2).astype(bool)
        new_grid_3d[:, :, :cols][~flipped_selection] = 0
        return new_grid_3d.astype(int)


    def duplicate_horizontally(self, grid, selection):
        """
        Duplicate the selection horizontally out of the original grid. Works only if columns <= 15.
        """
        d, rows, cols = np.shape(selection)
        
        if cols > 15:
            return create_grid3d(grid, selection)  # Return 3D grid form if columns exceed the limit
        
        grid_3d = create_grid3d(grid, selection)
        new_grid_3d = np.zeros((d, rows, cols * 2))
        
        # Copy the original grid and duplicate the selected part horizontally
        new_grid_3d[:, :, :cols] = grid_3d
        new_grid_3d[:, :, cols:][selection.astype(bool)] = grid_3d[selection.astype(bool)]
        
        return new_grid_3d


    def duplicate_vertically(self, grid, selection):
        """
        Duplicate the selection vertically out of the original grid. Works only if rows <= 15.
        """
        d, rows, cols = np.shape(selection)
        
        if rows > 15:
            return create_grid3d(grid, selection)  # Return 3D grid form if rows exceed the limit
        
        grid_3d = create_grid3d(grid, selection)
        new_grid_3d = np.zeros((d, rows * 2, cols))
        
        # Copy the original grid and duplicate the selected part vertically
        new_grid_3d[:, :rows, :] = grid_3d
        new_grid_3d[:, rows:, :][selection.astype(bool)] = grid_3d[selection.astype(bool)]
        
        return new_grid_3d
   
    # Copy-paste transformations
    def copy_paste_vertically(self, grid, selection):
        """
        For each mask in the selection, copy its selected area and paste it upwards and downwards
        as many times as possible within the grid bounds.
        """
        grid_3d = create_grid3d(grid, selection)
        n_masks, height_of_grid, width_of_grid = grid_3d.shape

        # Identify rows with at least one '1' in each mask
        rows_with_one = np.any(selection == 1, axis=2)  # Shape: (n_masks, rows)

        # Initialize arrays for first and last rows containing '1's
        first_rows = np.full(n_masks, -1)
        last_rows = np.full(n_masks, -1)

        # Find first and last rows with '1's in each mask
        for idx in range(n_masks):
            row_indices = np.where(rows_with_one[idx])[0]
            if row_indices.size > 0:
                first_rows[idx] = row_indices[0]
                last_rows[idx] = row_indices[-1]

        # Calculate the height of the selection per mask
        selection_height = last_rows - first_rows + 1  # Shape: (n_masks,)
        # Calculate factors per mask
        factor_up = np.ceil(first_rows / selection_height).astype(int)
        factor_down = np.ceil((height_of_grid - last_rows - 1) / selection_height).astype(int)
        # Initialize the final transformation
        final_transformation = grid_3d.copy()

        # Loop over each mask
        for idx in range(n_masks):
            if selection_height[idx] <= 0:
                # Skip masks with no selection
                continue

            # Get the grid and selection for the current mask
            grid_layer = final_transformation[idx]
            selection_layer = selection[idx]
            # Reshape to 3D arrays to match the input of copy_paste
            grid_layer_3d = np.expand_dims(grid_layer, axis=0)
            selection_layer_3d = np.expand_dims(selection_layer, axis=0)

            # Copy-paste upwards
            for i in range(factor_up[idx]):
                shift = -(i+1) * selection_height[idx]
                # Perform the copy-paste
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, 0, shift) #todo verify this part
            # Copy-paste downwards
            for i in range(factor_down[idx]):
                shift = (i+1) * selection_height[idx]
                # Perform the copy-paste
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, 0, shift)

            # Remove the extra dimension and update the final transformation
            final_transformation[idx] = grid_layer_3d[0]

        return final_transformation

    def copy_paste_horizontally(self, grid, selection):
        """
        For each mask in the selection, copy its selected area and paste it leftwards and rightwards
        as many times as possible within the grid bounds.
        """
        grid_3d = create_grid3d(grid, selection)
        n_masks, height_of_grid, width_of_grid = grid_3d.shape
    
        # Identify columns with at least one '1' in each mask
        columns_with_one = np.any(selection == 1, axis=1)  # Shape: (n_masks, columns)
    
        # Initialize arrays for first and last columns containing '1's
        first_cols = np.full(n_masks, -1)
        last_cols = np.full(n_masks, -1)
    
        # Find first and last columns with '1's in each mask
        for idx in range(n_masks):
            col_indices = np.where(columns_with_one[idx])[0]
            if col_indices.size > 0:
                first_cols[idx] = col_indices[0]
                last_cols[idx] = col_indices[-1]
    
        # Calculate the width of the selection per mask
        selection_width = last_cols - first_cols + 1  # Shape: (n_masks,)
        # Calculate factors per mask
        factor_left = np.ceil(first_cols / selection_width).astype(int)
        factor_right = np.ceil((width_of_grid - last_cols - 1) / selection_width).astype(int)
        # Initialize the final transformation
        final_transformation = grid_3d.copy()
    
        # Loop over each mask
        for idx in range(n_masks):
            if selection_width[idx] <= 0:
                # Skip masks with no selection
                continue
    
            # Get the grid and selection for the current mask
            grid_layer = final_transformation[idx]
            selection_layer = selection[idx]
            # Reshape to 3D arrays to match the input of copy_paste
            grid_layer_3d = np.expand_dims(grid_layer, axis=0)
            selection_layer_3d = np.expand_dims(selection_layer, axis=0)
    
            # Copy-paste leftwards
            for i in range(factor_left[idx]):
                shift = -(i + 1) * selection_width[idx]
                # Perform the copy-paste
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, shift, 0)
            # Copy-paste rightwards
            for i in range(factor_right[idx]):
                shift = (i + 1) * selection_width[idx]
                # Perform the copy-paste
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, shift, 0)
    
            # Remove the extra dimension and update the final transformation
            final_transformation[idx] = grid_layer_3d[0]
    
        return final_transformation
        
    def copy_paste(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops.
        """
        grid_3d = create_grid3d(grid, selection)

        # Get the indices where the selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Compute the new coordinates after shifting
        new_row_idxs = old_row_idxs + shift_y  # Shift rows (vertical)
        new_col_idxs = old_col_idxs + shift_x  # Shift columns (horizontal)

        # Filter out coordinates that are out of bounds
        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        # Apply the valid mask to indices and coordinates
        layer_idxs = layer_idxs[valid_mask]
        old_row_idxs = old_row_idxs[valid_mask]
        old_col_idxs = old_col_idxs[valid_mask]
        new_row_idxs = new_row_idxs[valid_mask]
        new_col_idxs = new_col_idxs[valid_mask]

        # Get the values to copy
        values = grid_3d[layer_idxs, old_row_idxs, old_col_idxs]

        # Copy the values to the new positions
        grid_3d[layer_idxs, new_row_idxs, new_col_idxs] = values

        return grid_3d
    
    def copy_sum(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops.
        """
        grid_3d = create_grid3d(grid, selection)

        # Get the indices where the selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Compute the new coordinates after shifting
        new_row_idxs = old_row_idxs + shift_y  # Shift rows (vertical)
        new_col_idxs = old_col_idxs + shift_x  # Shift columns (horizontal)

        # Filter out coordinates that are out of bounds
        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        # Apply the valid mask to indices and coordinates
        layer_idxs = layer_idxs[valid_mask]
        old_row_idxs = old_row_idxs[valid_mask]
        old_col_idxs = old_col_idxs[valid_mask]
        new_row_idxs = new_row_idxs[valid_mask]
        new_col_idxs = new_col_idxs[valid_mask]

        # Get the values to copy
        values = grid_3d[layer_idxs, old_row_idxs, old_col_idxs]

        # Copy the values to the new positions
        np.add.at(grid_3d, (layer_idxs, new_row_idxs, new_col_idxs), values)

        # use modulo to keep the values in the range [0, 9]
        grid_3d = grid_3d % 10

        return grid_3d
    
    def cut_paste(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops.
        """
        grid_3d = create_grid3d(grid, selection)

        # Get the indices where the selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Compute the new coordinates after shifting
        new_row_idxs = old_row_idxs + shift_y  # Shift rows (vertical)
        new_col_idxs = old_col_idxs + shift_x  # Shift columns (horizontal)

        # Filter out coordinates that are out of bounds
        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        # Get the values to move
        values = grid_3d[layer_idxs[valid_mask], old_row_idxs[valid_mask], old_col_idxs[valid_mask]]

        # Clear the original positions
        grid_3d[layer_idxs, old_row_idxs, old_col_idxs] = 0

        # Assign the values to the new positions
        grid_3d[layer_idxs[valid_mask], new_row_idxs[valid_mask], new_col_idxs[valid_mask]] = values

        return grid_3d

    def cut_sum(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops.
        """
        grid_3d = create_grid3d(grid, selection)

        # Get the indices where the selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Compute the new coordinates after shifting
        new_row_idxs = old_row_idxs + shift_y  # Shift rows (vertical)
        new_col_idxs = old_col_idxs + shift_x  # Shift columns (horizontal)

        # Filter out coordinates that are out of bounds
        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        # Get the values to move
        values = grid_3d[layer_idxs[valid_mask], old_row_idxs[valid_mask], old_col_idxs[valid_mask]]

        # Clear the original positions
        grid_3d[layer_idxs, old_row_idxs, old_col_idxs] = 0

        # Paste the values to the new positions adding them to the existing values
        np.add.at(grid_3d, (layer_idxs[valid_mask], new_row_idxs[valid_mask], new_col_idxs[valid_mask]), values)

        # use modulo to keep the values in the range [0, 9]
        grid_3d = grid_3d % 10

        return grid_3d
    

    # Gravitate transformations
    def gravitate_whole_downwards_paste(self, grid, selection):
        """
        Copies and pastes the selected cells in the grid downwards until they reach either
        the bottom of the grid or they land on top of non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape
        
        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()

        # Get the indices where selection is True
        indices = np.nonzero(selection)  # This returns a tuple of arrays (depth_indices, row_indices, col_indices)

        # Zero out only the corresponding positions in grid_without_selection
        grid_without_selection[indices] = 0

        # Create row indices
        row_indices = np.arange(rows).reshape(1, rows, 1)  # Shape: (1, rows, 1)

        # Compute the maximum row index of the selection per depth and column
        sel_row_indices = np.where(selection, row_indices, -1)  # Shape: (depth, rows, cols)
        max_row_sel = sel_row_indices.max(axis=1)  # Shape: (depth, cols)

        # Identify columns with selection
        selection_exists_in_column = (max_row_sel != -1)  # Shape: (depth, cols)

        # Expand dimensions for broadcasting
        max_row_sel_expanded = max_row_sel[:, None, :]  # Shape: (depth, 1, cols)

        # Create a mask for positions below the selection
        mask_below_selection = row_indices > max_row_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles below the selection
        obstacles_below = (grid_without_selection != 0) & mask_below_selection  # Shape: (depth, rows, cols)

        # Determine the first obstacle row index per depth and column
        obstacle_positions = np.where(obstacles_below, row_indices, rows)

        # Replace obstacle positions in columns without selection with rows + 1
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, rows + 1)

        # Calculate the shift distance per depth and column
        shift_per_column = obstacle_positions.min(axis=1) - max_row_sel - 1  # Shape: (depth, cols)

        # In columns without selection, set shift_per_column to a large number
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)

        # Compute the minimum shift over columns with selection for each depth
        shift_per_depth = np.min(shift_per_column, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, rows)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth
        shift_y = shift_per_depth[layer_idxs]

        # Call cut_paste with shift_y as an array
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)
        
        return grid_3d
   
    def gravitate_whole_upwards_paste(self, grid, selection):
        """
        Copies and pastes the selected cells in the grid upwards as a whole until they reach either
        the top of the grid or they land under non-zero cells.
        """

        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape

        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        # Create row indices
        row_indices = np.arange(rows).reshape(1, rows, 1)  # Shape: (1, rows, 1)

        # Compute the minimum row index of the selection per depth and column
        sel_row_indices = np.where(selection, row_indices, rows)  # Shape: (depth, rows, cols)
        min_row_sel = sel_row_indices.min(axis=1)  # Shape: (depth, cols)

        # Identify columns with selection
        selection_exists_in_column = (min_row_sel != rows)  # Shape: (depth, cols)

        # Expand dimensions for broadcasting
        min_row_sel_expanded = min_row_sel[:, None, :]  # Shape: (depth, 1, cols)

        # Create a mask for positions above the selection
        mask_above_selection = row_indices < min_row_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles above the selection
        obstacles_above = (grid_without_selection != 0) & mask_above_selection  # Shape: (depth, rows, cols)

        # Determine the last obstacle row index per depth and column
        obstacle_positions = np.where(obstacles_above, row_indices, -1)

        # Replace obstacle positions in columns without selection with -1
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, -1)

        # Calculate the shift distance per depth and column
        shift_per_column = min_row_sel - obstacle_positions.max(axis=1) - 1  # Shape: (depth, cols)

        # In columns without selection, set shift_per_column to a large number
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)

        # Compute the minimum shift over columns with selection for each depth
        shift_per_depth = np.min(shift_per_column, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, rows).astype(int)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth (negative for upwards movement)
        shift_y = -shift_per_depth[layer_idxs]

        # Call cut_paste with shift_y as an array
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)

        return grid_3d
    
    def gravitate_whole_right_paste(self, grid, selection):
        """
        Copies and pastes the selected cells in the grid to the right as a whole until they reach either
        the right edge of the grid or they land next to non-zero cells.
        """

        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape

        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        # Create column indices
        col_indices = np.arange(cols).reshape(1, 1, cols)  # Shape: (1, 1, cols)

        # Compute the maximum column index of the selection per depth and row
        sel_col_indices = np.where(selection, col_indices, -1)  # Shape: (depth, rows, cols)
        max_col_sel = sel_col_indices.max(axis=2)  # Shape: (depth, rows)

        # Identify rows with selection
        selection_exists_in_row = (max_col_sel != -1)  # Shape: (depth, rows)

        # Expand dimensions for broadcasting
        max_col_sel_expanded = max_col_sel[:, :, None]  # Shape: (depth, rows, 1)

        # Create a mask for positions to the right of the selection
        mask_right_selection = col_indices > max_col_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles to the right of the selection
        obstacles_right = (grid_without_selection != 0) & mask_right_selection  # Shape: (depth, rows, cols)

        # Determine the first obstacle column index per depth and row
        obstacle_positions = np.where(obstacles_right, col_indices, cols)
        first_obstacle_col = obstacle_positions.min(axis=2)  # Shape: (depth, rows)

        # Replace obstacle positions in rows without selection with cols + 1
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, cols + 1)

        # Calculate the shift distance per depth and row
        shift_per_row = first_obstacle_col - max_col_sel - 1  # Shape: (depth, rows)

        # In rows without selection, set shift_per_row to a large number
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)

        # Compute the minimum shift over rows with selection for each depth
        shift_per_depth = np.min(shift_per_row, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth
        shift_x = shift_per_depth[layer_idxs]

        # Call cut_paste with shift_x as an array
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)

        return grid_3d
    
    def gravitate_whole_left_paste(self, grid, selection):
        """
        Copies and pastes the selected cells in the grid to the left as a whole until they reach either
        the left edge of the grid or they land next to non-zero cells.
        """
        import numpy as np

        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape

        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        # Create column indices
        col_indices = np.arange(cols).reshape(1, 1, cols)  # Shape: (1, 1, cols)

        # Compute the minimum column index of the selection per depth and row
        sel_col_indices = np.where(selection, col_indices, cols)  # Shape: (depth, rows, cols)
        min_col_sel = sel_col_indices.min(axis=2)  # Shape: (depth, rows)

        # Identify rows with selection
        selection_exists_in_row = (min_col_sel != cols)  # Shape: (depth, rows)

        # Expand dimensions for broadcasting
        min_col_sel_expanded = min_col_sel[:, :, None]  # Shape: (depth, rows, 1)

        # Create a mask for positions to the left of the selection
        mask_left_selection = col_indices < min_col_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles to the left of the selection
        obstacles_left = (grid_without_selection != 0) & mask_left_selection  # Shape: (depth, rows, cols)

        # Determine the last obstacle column index per depth and row
        obstacle_positions = np.where(obstacles_left, col_indices, -1)
        last_obstacle_col = obstacle_positions.max(axis=2)  # Shape: (depth, rows)

        # Replace obstacle positions in rows without selection with -1
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, -1)

        # Calculate the shift distance per depth and row
        shift_per_row = min_col_sel - last_obstacle_col - 1  # Shape: (depth, rows)

        # In rows without selection, set shift_per_row to a large number
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)

        # Compute the minimum shift over rows with selection for each depth
        shift_per_depth = np.min(shift_per_row, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth (negative for leftward movement)
        shift_x = -shift_per_depth[layer_idxs]

        # Call cut_paste with shift_x as an array
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)

        return grid_3d

    def gravitate_whole_downwards_cut(self, grid, selection):
        """
        Shift the selected cells in the grid downwards until they reach either
        the bottom of the grid or they land on top of non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape
        
        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()

        # Get the indices where selection is True
        indices = np.nonzero(selection)  # This returns a tuple of arrays (depth_indices, row_indices, col_indices)

        # Zero out only the corresponding positions in grid_without_selection
        grid_without_selection[indices] = 0

        # Create row indices
        row_indices = np.arange(rows).reshape(1, rows, 1)  # Shape: (1, rows, 1)

        # Compute the maximum row index of the selection per depth and column
        sel_row_indices = np.where(selection, row_indices, -1)  # Shape: (depth, rows, cols)
        max_row_sel = sel_row_indices.max(axis=1)  # Shape: (depth, cols)

        # Identify columns with selection
        selection_exists_in_column = (max_row_sel != -1)  # Shape: (depth, cols)

        # Expand dimensions for broadcasting
        max_row_sel_expanded = max_row_sel[:, None, :]  # Shape: (depth, 1, cols)

        # Create a mask for positions below the selection
        mask_below_selection = row_indices > max_row_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles below the selection
        obstacles_below = (grid_without_selection != 0) & mask_below_selection  # Shape: (depth, rows, cols)

        # Determine the first obstacle row index per depth and column
        obstacle_positions = np.where(obstacles_below, row_indices, rows)

        # Replace obstacle positions in columns without selection with rows + 1
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, rows + 1)

        # Calculate the shift distance per depth and column
        shift_per_column = obstacle_positions.min(axis=1) - max_row_sel - 1  # Shape: (depth, cols)

        # In columns without selection, set shift_per_column to a large number
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)

        # Compute the minimum shift over columns with selection for each depth
        shift_per_depth = np.min(shift_per_column, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, rows)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth
        shift_y = shift_per_depth[layer_idxs]

        # Call cut_paste with shift_y as an array
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)
        
        return grid_3d

    def gravitate_whole_upwards_cut(self, grid, selection):
        """
        Shift the selected cells in the grid upwards as a whole until they reach either
        the top of the grid or they land under non-zero cells.
        """

        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape

        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        # Create row indices
        row_indices = np.arange(rows).reshape(1, rows, 1)  # Shape: (1, rows, 1)

        # Compute the minimum row index of the selection per depth and column
        sel_row_indices = np.where(selection, row_indices, rows)  # Shape: (depth, rows, cols)
        min_row_sel = sel_row_indices.min(axis=1)  # Shape: (depth, cols)

        # Identify columns with selection
        selection_exists_in_column = (min_row_sel != rows)  # Shape: (depth, cols)

        # Expand dimensions for broadcasting
        min_row_sel_expanded = min_row_sel[:, None, :]  # Shape: (depth, 1, cols)

        # Create a mask for positions above the selection
        mask_above_selection = row_indices < min_row_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles above the selection
        obstacles_above = (grid_without_selection != 0) & mask_above_selection  # Shape: (depth, rows, cols)

        # Determine the last obstacle row index per depth and column
        obstacle_positions = np.where(obstacles_above, row_indices, -1)

        # Replace obstacle positions in columns without selection with -1
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, -1)

        # Calculate the shift distance per depth and column
        shift_per_column = min_row_sel - obstacle_positions.max(axis=1) - 1  # Shape: (depth, cols)

        # In columns without selection, set shift_per_column to a large number
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)

        # Compute the minimum shift over columns with selection for each depth
        shift_per_depth = np.min(shift_per_column, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, rows).astype(int)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth (negative for upwards movement)
        shift_y = -shift_per_depth[layer_idxs]

        # Call cut_paste with shift_y as an array
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)

        return grid_3d
 
    def gravitate_whole_right_cut(self, grid, selection):
        """
        Shift the selected cells in the grid to the right as a whole until they reach either
        the right edge of the grid or they land next to non-zero cells.
        """

        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape

        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        # Create column indices
        col_indices = np.arange(cols).reshape(1, 1, cols)  # Shape: (1, 1, cols)

        # Compute the maximum column index of the selection per depth and row
        sel_col_indices = np.where(selection, col_indices, -1)  # Shape: (depth, rows, cols)
        max_col_sel = sel_col_indices.max(axis=2)  # Shape: (depth, rows)

        # Identify rows with selection
        selection_exists_in_row = (max_col_sel != -1)  # Shape: (depth, rows)

        # Expand dimensions for broadcasting
        max_col_sel_expanded = max_col_sel[:, :, None]  # Shape: (depth, rows, 1)

        # Create a mask for positions to the right of the selection
        mask_right_selection = col_indices > max_col_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles to the right of the selection
        obstacles_right = (grid_without_selection != 0) & mask_right_selection  # Shape: (depth, rows, cols)

        # Determine the first obstacle column index per depth and row
        obstacle_positions = np.where(obstacles_right, col_indices, cols)
        first_obstacle_col = obstacle_positions.min(axis=2)  # Shape: (depth, rows)

        # Replace obstacle positions in rows without selection with cols + 1
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, cols + 1)

        # Calculate the shift distance per depth and row
        shift_per_row = first_obstacle_col - max_col_sel - 1  # Shape: (depth, rows)

        # In rows without selection, set shift_per_row to a large number
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)

        # Compute the minimum shift over rows with selection for each depth
        shift_per_depth = np.min(shift_per_row, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth
        shift_x = shift_per_depth[layer_idxs]

        # Call cut_paste with shift_x as an array
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)

        return grid_3d
    
    def gravitate_whole_left_cut(self, grid, selection):
        """
        Shift the selected cells in the grid to the left as a whole until they reach either
        the left edge of the grid or they land next to non-zero cells.
        """
        import numpy as np

        grid_3d = create_grid3d(grid, selection)

        # Get dimensions
        depth, rows, cols = selection.shape

        # Exclude the selection from the grid to avoid self-collision
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        # Create column indices
        col_indices = np.arange(cols).reshape(1, 1, cols)  # Shape: (1, 1, cols)

        # Compute the minimum column index of the selection per depth and row
        sel_col_indices = np.where(selection, col_indices, cols)  # Shape: (depth, rows, cols)
        min_col_sel = sel_col_indices.min(axis=2)  # Shape: (depth, rows)

        # Identify rows with selection
        selection_exists_in_row = (min_col_sel != cols)  # Shape: (depth, rows)

        # Expand dimensions for broadcasting
        min_col_sel_expanded = min_col_sel[:, :, None]  # Shape: (depth, rows, 1)

        # Create a mask for positions to the left of the selection
        mask_left_selection = col_indices < min_col_sel_expanded  # Shape: (depth, rows, cols)

        # Identify obstacles to the left of the selection
        obstacles_left = (grid_without_selection != 0) & mask_left_selection  # Shape: (depth, rows, cols)

        # Determine the last obstacle column index per depth and row
        obstacle_positions = np.where(obstacles_left, col_indices, -1)
        last_obstacle_col = obstacle_positions.max(axis=2)  # Shape: (depth, rows)

        # Replace obstacle positions in rows without selection with -1
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, -1)

        # Calculate the shift distance per depth and row
        shift_per_row = min_col_sel - last_obstacle_col - 1  # Shape: (depth, rows)

        # In rows without selection, set shift_per_row to a large number
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)

        # Compute the minimum shift over rows with selection for each depth
        shift_per_depth = np.min(shift_per_row, axis=1)  # Shape: (depth,)

        # Ensure non-negative shifts
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        # Get indices where selection is True
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        # Get the shift for each selected cell based on its depth (negative for leftward movement)
        shift_x = -shift_per_depth[layer_idxs]

        # Call cut_paste with shift_x as an array
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)

        return grid_3d
    
    def down_gravity(self, grid, selection):
        """
        Apply gravity to selected cells, moving them down until they hit non-zero cells or the bottom of the grid.
        """

        # Step 1: Convert to 3D grid using the selection and grid
        grid_3d = create_grid3d(grid, selection)

        # Step 2: Get dimensions of the grid
        num_layers, num_rows, num_cols = grid_3d.shape
        
        # Step 3: Process each selection layer individually
        for layer_idx in range(num_layers):
            # Extract the current selection mask (2D)
            selection_layer = selection[layer_idx]
            
            # Create a mask where the cells are selected (non-zero)
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells
            
            # Step 4: Process each selected cell, starting from the bottom
            for i in range(len(selected_rows)-1, -1, -1):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                
                # Clear the current position in the grid and the selection mask
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                # Step 5: Find the target position by moving down
                for target_row in range(row + 1, num_rows):
                    if grid_3d[layer_idx, target_row, col] != 0:  # Hit non-zero cell
                        grid_3d[layer_idx, target_row - 1, col] = value
                        selection_layer[target_row - 1, col] = 1
                        break
                else:
                    # No non-zero cell encountered, drop to the bottom row
                    grid_3d[layer_idx, num_rows - 1, col] = value
                    selection_layer[num_rows - 1, col] = 1

        return grid_3d
    
    def up_gravity(self, grid, selection):
        """
        Apply upward gravity to selected cells, moving them up until they hit non-zero cells or the top of the grid.
        The function vectorizes this operation by using the create_grid3d function.
        """

        # Step 1: Convert to 3D grid using the selection and grid
        grid_3d = create_grid3d(grid, selection)

        # Step 2: Get dimensions of the grid
        num_layers, num_rows, num_cols = grid_3d.shape
        
        # Step 3: Process each selection layer individually
        for layer_idx in range(num_layers):
            # Extract the current selection mask (2D)
            selection_layer = selection[layer_idx]
            
            # Create a mask where the cells are selected (non-zero)
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells
            
            # Step 4: Process each selected cell, starting from the top
            for i in range(len(selected_rows)):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                
                # Clear the current position in the grid and the selection mask
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                # Step 5: Find the target position by moving up
                for target_row in range(row - 1, -1, -1):
                    if grid_3d[layer_idx, target_row, col] != 0:  # Hit non-zero cell
                        grid_3d[layer_idx, target_row + 1, col] = value
                        selection_layer[target_row + 1, col] = 1
                        break
                else:
                    # No non-zero cell encountered, drop to the top row
                    grid_3d[layer_idx, 0, col] = value
                    selection_layer[0, col] = 1

        return grid_3d

    def right_gravity(self, grid, selection):
        """
        Apply gravity to selected cells, moving them towards the right until they hit non-zero cells or the rightmost column.
        The function vectorizes this operation by using the create_grid3d function.
        """

        # Step 1: Convert to 3D grid using the selection and grid
        grid_3d = create_grid3d(grid, selection)

        # Step 2: Get dimensions of the grid
        num_layers, num_rows, num_cols = grid_3d.shape
        
        # Step 3: Process each selection layer individually
        for layer_idx in range(num_layers):
            # Extract the current selection mask (2D)
            selection_layer = selection[layer_idx]
            
            # Create a mask where the cells are selected (non-zero)
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells
            
            # Step 4: Process each selected cell, starting from the right
            for i in range(len(selected_cols)-1, -1, -1):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                
                # Clear the current position in the grid and the selection mask
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                # Step 5: Find the target position by moving right
                for target_col in range(col + 1, num_cols):
                    if grid_3d[layer_idx, row, target_col] != 0:  # Hit non-zero cell
                        grid_3d[layer_idx, row, target_col - 1] = value
                        selection_layer[row, target_col - 1] = 1
                        break
                else:
                    # No non-zero cell encountered, move to the far right column
                    grid_3d[layer_idx, row, num_cols - 1] = value
                    selection_layer[row, num_cols - 1] = 1

        return grid_3d

    def left_gravity(self, grid, selection):
        """
        Apply gravity to selected cells, moving them towards the left until they hit non-zero cells or the leftmost column.
        The function vectorizes this operation by using the create_grid3d function.
        """

        # Step 1: Convert to 3D grid using the selection and grid
        grid_3d = create_grid3d(grid, selection)

        # Step 2: Get dimensions of the grid
        num_layers, num_rows, num_cols = grid_3d.shape
        
        # Step 3: Process each selection layer individually
        for layer_idx in range(num_layers):
            # Extract the current selection mask (2D)
            selection_layer = selection[layer_idx]
            
            # Create a mask where the cells are selected (non-zero)
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells
            
            # Step 4: Process each selected cell, starting from the left
            for i in range(len(selected_cols)-1, -1, -1):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                
                # Clear the current position in the grid and the selection mask
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                # Step 5: Find the target position by moving left
                for target_col in range(col - 1, -1, -1):
                    if grid_3d[layer_idx, row, target_col] != 0:  # Hit non-zero cell
                        grid_3d[layer_idx, row, target_col + 1] = value
                        selection_layer[row, target_col + 1] = 1
                        break
                else:
                    # No non-zero cell encountered, move to the far left column
                    grid_3d[layer_idx, row, 0] = value
                    selection_layer[row, 0] = 1

        return grid_3d


    # Upscale transformations
    def vupscale(self, grid, selection, scale_factor):
        """
        Upscale the selection in the grid by a specified scale factor, 
        and cap the upscaled selection to match the original size.
        """
        # Create a 3D grid representation
        selection_3d_grid = create_grid3d(grid, selection)
        depth, original_rows, original_cols = np.shape(selection)

        # Perform upscaling by repeating elements along rows
        upscaled_selection = np.repeat(selection, scale_factor, axis=1)
        upscaled_selection_3d_grid = np.repeat(selection_3d_grid, scale_factor, axis=1)

        # Calculate row boundaries for capping
        if original_rows % 2 == 0:
            half_rows_top, half_rows_bottom = original_rows // 2, original_rows // 2
        else:
            half_rows_top, half_rows_bottom = original_rows // 2 + 1, original_rows // 2

        # Initialize arrays for capped selection and grid
        capped_selection = np.zeros((depth, original_rows, original_cols), dtype=bool)
        capped_upscaled_grid = np.zeros((depth, original_rows, original_cols))

        for layer_idx in range(depth):
            # Compute center of mass for the original and upscaled selection
            original_com = center_of_mass(selection[layer_idx])[0]
            upscaled_com = center_of_mass(upscaled_selection[layer_idx])[0]

            # Determine bounds for capping
            lower_bound = min(int(upscaled_com + half_rows_bottom), original_rows * scale_factor)
            upper_bound = max(int(upscaled_com - half_rows_top), 0)

            # Adjust bounds if out of range
            if lower_bound >= original_rows * scale_factor:
                lower_bound = original_rows * scale_factor
                upper_bound = lower_bound - original_rows
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_rows

            # Apply capping and recalculate center of mass
            capped_selection[layer_idx] = upscaled_selection[layer_idx, upper_bound:lower_bound, :]
            capped_com = center_of_mass(capped_selection[layer_idx])[0]

            # Adjust bounds based on center of mass difference
            offset = capped_com - original_com
            lower_bound += offset
            upper_bound += offset

            # Reapply bounds check
            if lower_bound >= original_rows * scale_factor:
                lower_bound = original_rows * scale_factor
                upper_bound = lower_bound - original_rows
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_rows

            # Final capping
            capped_selection[layer_idx] = upscaled_selection[layer_idx, upper_bound:lower_bound, :]
            capped_upscaled_grid[layer_idx] = upscaled_selection_3d_grid[layer_idx, upper_bound:lower_bound, :]

        # Update the original grid with the capped selection
        selection_3d_grid[selection == 1] = 0
        selection_3d_grid[capped_selection] = capped_upscaled_grid[capped_selection].ravel()

        return selection_3d_grid
 
    def hupscale(self, grid, selection, scale_factor):
        """
        Upscale the selection in the grid horizontally by a specified scale factor,
        and cap the upscaled selection to match the original size.
        """
        # Create a 3D grid representation
        selection_3d_grid = create_grid3d(grid, selection)
        depth, original_rows, original_cols = selection.shape

        # Perform upscaling by repeating elements along columns
        upscaled_selection = np.repeat(selection, scale_factor, axis=2)
        upscaled_selection_3d_grid = np.repeat(selection_3d_grid, scale_factor, axis=2)
        upscaled_cols = upscaled_selection.shape[2]

        # Calculate column boundaries for capping
        if original_cols % 2 == 0:
            half_cols_left, half_cols_right = original_cols // 2, original_cols // 2
        else:
            half_cols_left, half_cols_right = original_cols // 2 + 1, original_cols // 2

        # Initialize arrays for capped selection and grid
        capped_selection = np.zeros((depth, original_rows, original_cols), dtype=bool)
        capped_upscaled_grid = np.zeros((depth, original_rows, original_cols))

        for layer_idx in range(depth):
            # Compute center of mass for the original and upscaled selection
            original_com = center_of_mass(selection[layer_idx])[1]
            upscaled_com = center_of_mass(upscaled_selection[layer_idx])[1]

            # Determine bounds for capping
            lower_bound = min(int(upscaled_com + half_cols_right), upscaled_cols)
            upper_bound = max(int(upscaled_com - half_cols_left), 0)

            # Adjust bounds if out of range
            if lower_bound >= upscaled_cols:
                lower_bound = upscaled_cols
                upper_bound = lower_bound - original_cols
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_cols

            # Apply capping and recalculate center of mass
            capped_selection[layer_idx] = upscaled_selection[layer_idx, :, upper_bound:lower_bound]
            capped_com = center_of_mass(capped_selection[layer_idx])[1]

            # Adjust bounds based on center of mass difference
            offset = int(capped_com - original_com)
            lower_bound += offset
            upper_bound += offset

            # Reapply bounds check
            if lower_bound >= upscaled_cols:
                lower_bound = upscaled_cols
                upper_bound = lower_bound - original_cols
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_cols

            # Final capping
            capped_selection[layer_idx] = upscaled_selection[layer_idx, :, upper_bound:lower_bound]
            capped_upscaled_grid[layer_idx] = upscaled_selection_3d_grid[layer_idx, :, upper_bound:lower_bound]

        # Update the original grid with the capped selection
        selection_3d_grid[selection == 1] = 0
        capped_mask = capped_selection.astype(bool)
        selection_3d_grid[capped_mask] = capped_upscaled_grid[capped_mask].ravel()

        return selection_3d_grid


    # Delete transformations
    def crop(self, grid, selection):
        """
        Crop the grid to the bounding rectangle around the selection. Use -1 as the value for cells outside the selection.
        -1 will be the same number that will be used to pad the grids in order to make them the same size.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)

        # Handle cases where the bounding rectangle for a specific selection is just made of False values
        for i in range(selection.shape[0]):  # Iterate over the batch dimension
            if not bounding_rectangle[i].any():  # Check if the rectangle is entirely False
                bounding_rectangle[i] = np.ones_like(grid_3d[i], dtype=bool)  # Mark the whole grid for this selection
        
        grid_3d[~bounding_rectangle] = -1
        return grid_3d
    
    def delete(self, grid, selection):
        """
        Set the value of the selected cells to 0.
        """
        grid_3d = create_grid3d(grid, selection)
        grid_3d[selection] = 0
        return grid_3d



    def change_background_color(self, grid, selection, new_color):
        '''
        Change the background color of the grid to the specified color.
        '''
    
        grid3d = create_grid3d(grid, selection)
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid) # Get the most common color in the grid
        background_color = int(background_color) # Convert the color to an integer
        grid3d[grid3d == background_color] = new_color # Change the background color to the specified color

        if check_color(new_color) == False: # Check if the color is valid
            return grid3d
        
        return grid3d
    
    def change_selection_to_background_color(self, grid, selection):
        '''
        Change the selected cells in the grid to the background color.
        ''' 
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid)
        grid_3d = create_grid3d(grid, selection)
        grid_3d[selection == 1] = background_color

        return grid_3d

    def vectorized_vupscale(self, grid, selection, scale_factor):
        """
        Upscale the selection in the grid vertically by a specified scale factor,
        and overwrite existing values
        """
        grid_3d = create_grid3d(grid, selection)
        depth, original_rows, original_cols = selection.shape

        # Upscale selection and grid
        upscaled_selection = np.repeat(selection, scale_factor, axis=1)
        upscaled_grid = np.repeat(grid_3d * selection, scale_factor, axis=1)
        upscaled_rows = upscaled_selection.shape[1]

        com_original = vectorized_center_of_mass(selection)
        
        # Compute centers of mass for the upscaled selection
        com_upscaled = vectorized_center_of_mass(upscaled_selection)   
        
        # Compute shift to align centers of mass
        shift = (com_original - com_upscaled )

        # Create shifted row indices for upscaled selection
        shifted_row_indices = np.arange(upscaled_rows).reshape(1, upscaled_rows, 1) + shift
        shifted_row_indices = np.broadcast_to(shifted_row_indices, upscaled_selection.shape)
        
        # Create a mask for valid shifted indices
        valid_mask = (
            (shifted_row_indices >= 0) &
            (shifted_row_indices < original_rows) &
            upscaled_selection
        )

        # Get indices where valid
        indices = np.argwhere(valid_mask)
        d = indices[:, 0]
        r_upscaled = indices[:, 1]
        c = indices[:, 2]
        shifted_rows = shifted_row_indices[valid_mask].flatten()
        values = upscaled_grid[valid_mask].flatten()

        # Reverse the order to overwrite values in case of overlaps
        rev_indices = np.arange(len(d) - 1, -1, -1)
        d_rev = d[rev_indices]
        shifted_rows_rev = shifted_rows[rev_indices]
        c_rev = c[rev_indices]
        values_rev = values[rev_indices]

        # Use indexing to assign values to the final grid
        final_grid = np.zeros((depth, original_rows, original_cols), dtype=grid_3d.dtype)
        final_grid[d_rev, shifted_rows_rev, c_rev] = values_rev

        # Clear the original selection from the grid
        grid_3d[selection] = 0

        # Overwrite the grid with the final grid
        grid_3d[final_grid != 0] = final_grid[final_grid != 0]

        return grid_3d
    
