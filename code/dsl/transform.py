import numpy as np
from dsl.utilities.plot import plot_selection
from dsl.utilities.checks import check_axis, check_num_rotations, check_color
from scipy.ndimage import binary_fill_holes
from dsl.utilities.transformation_utilities import create_grid3d, find_bounding_rectangle, find_bounding_square, center_of_mass
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
# 13 - cut_paste(grid, selection, shift_x, shift_y): Shift the selected cells in the grid by (shift_x, shift_y) and set the original cells to 0.
# 14 - change_background_color(grid, selection, new_color): Change the background color of the grid to the specified color.
# 15 - vupscale(grid, selection, scale_factor): Upscale the selection in the grid by a specified scale factor, and cap the upscaled selection to match the original size.

class Transformer:
    def __init__(self):
        pass

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
    
    def rotate90(self, grid, selection):
        """
        Rotate the selected cells 90 degrees counterclockwise.
        """
        return self.rotate(grid, selection, 1)

    def rotate180(self, grid, selection):
        """
        Rotate the selected cells 180 degrees counterclockwise.
        """
        return self.rotate(grid, selection, 2)
    
    def rotate270(self, grid, selection):
        """
        Rotate the selected cells 270 degrees counterclockwise.
        """
        return self.rotate(grid, selection, 3)
    
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
        # TODO @vittorio (credo) : La funzione deve prendere la selection come input e non fare la selezione da sola
        # TODO @vittorio : controllare se il colore è nel range dei colori possibili, altrimenti ritornare la grid_3d invariata
        selector = Selector(grid.shape)
        bounding_shapes = selector.select_colored_separated_shapes(grid, color)  # Get all separated shapes

        # Combine all bounding shapes into one mask
        combined_mask = np.any(bounding_shapes, axis=0)

        # Detect holes inside the combined mask
        filled_mask = binary_fill_holes(combined_mask)  # Fill all enclosed regions within the combined mask

        # Create a new grid with the filled mask
        filled_grid = grid.copy()
        filled_grid[filled_mask] = fill_color  # Fill the entire bounding shape (and its holes)

        # Ensure original color (`3`) remains in the bounding region
        filled_grid[combined_mask] = color

        return filled_grid
    
    def mirror_main_diagonal(self, grid, selection):
        '''
        Mirror the selected region along the main diagonal (top-left to bottom-right).
        '''
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)  # Find the bounding square for each selection slice

        for i in range(grid_3d.shape[0]):  # Iterate through each selection slice
            mask = bounding_square[i]  # Mask for the current bounding square
            rows, cols = np.where(mask)  # Get the indices of the selected region
            if len(rows) > 0 and len(cols) > 0:
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

    def mirror_anti_diagonal(self, grid, selection):
        '''
        Mirror the selected region along the anti-diagonal (top-right to bottom-left).
        '''
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)  # Find the bounding square for each selection slice

        for i in range(grid_3d.shape[0]):  # Iterate through each selection slice
            mask = bounding_square[i]  # Mask for the current bounding square
            rows, cols = np.where(mask)  # Get the indices of the selected region
            if len(rows) > 0 and len(cols) > 0:
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
    
    def copy_paste(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y).
        """
        grid_3d = create_grid3d(grid, selection)
        # Extract the selected values #keeping the original color values
        selected_values = grid_3d*selection

        # TODO @vittorio : controllare se shift_x e shift_y sono nel range delle dimensioni della griglia, altrimenti ritornare la grid_3d invariata
                
        # For each layer
        for idx in range(selection.shape[0]):
            layer_selection = selection[idx]
            coords = np.argwhere(layer_selection)
            # Add shift to coordinates
            new_coords = coords + np.array([shift_x, shift_y])
            # Filter out coordinates that are out of bounds
            valid_indices = (new_coords[:,0] >= 0) & (new_coords[:,0] < grid_3d.shape[1]) & \
                            (new_coords[:,1] >= 0) & (new_coords[:,1] < grid_3d.shape[2])
            coords = coords[valid_indices]
            new_coords = new_coords[valid_indices]
            # Paste the cut selection to the new positions
            for (old_i, old_j), (new_i, new_j) in zip(coords, new_coords):
                grid_3d[idx, new_i, new_j] = selected_values[idx, old_i, old_j]
        
        return grid_3d
    
    def cut_paste(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y).
        """
        grid_3d = create_grid3d(grid, selection)
        grid_3d_o = grid_3d.copy()
        # TODO @vittorio : controllare se shift_x e shift_y sono nel range delle dimensioni della griglia, altrimenti ritornare la grid_3d invariata
        
        # Extract the selected values #keeping the original color values
        selected_values = grid_3d*selection
                
        # For each layer
        for idx in range(selection.shape[0]):
            layer_selection = selection[idx]
            coords = np.argwhere(layer_selection)
            # Add shift to coordinates
            new_coords = coords + np.array([shift_x, shift_y])
            # Filter out coordinates that are out of bounds
            valid_indices = (new_coords[:,0] >= 0) & (new_coords[:,0] < grid_3d.shape[1]) & \
                            (new_coords[:,1] >= 0) & (new_coords[:,1] < grid_3d.shape[2])
            coords = coords[valid_indices]
            new_coords = new_coords[valid_indices]
            # Paste the cut selection to the new positions
            for (old_i, old_j), (new_i, new_j) in zip(coords, new_coords):
                grid_3d[idx, new_i, new_j] = selected_values[idx, old_i, old_j]
        
        grid_3d_f = - grid_3d_o + grid_3d

        return grid_3d_f
    
    def change_background_color(self, grid, selection, new_color):
        '''
        Change the background color of the grid to the specified color.
        '''
        # TODO @vittorio : controllare se new_color è nel range dei colori possibili, altrimenti ritornare la grid_3d invariata
        grid_f_w = grid.copy()
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid)
        grid_f_w[grid_f_w == background_color] = new_color
        grid3d = create_grid3d(grid_f_w,selection)
        
        return grid3d
    
    def change_selection_to_background_color(self, grid, selection):
        '''
        Change the selected cells in the grid to the background color.
        ''' 
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid)
        grid_3d = create_grid3d(grid, selection)
        for idx in range(selection.shape[0]):
            grid_3d[idx][selection[idx] == 1] = background_color

        return grid_3d

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
        selection_3d_grid[selection] = 0
        selection_3d_grid[capped_selection] = capped_upscaled_grid[capped_selection].ravel()

        return selection_3d_grid
    
    def hupscale(self, grid, selection, scale_factor):
        # TODO @filippo : Implement horizontal upscaling

        pass

    def fill_bounding_rectangle_with_color(self, grid, selection, color):
        '''
        Fill the bounding rectangle around the selection with the specified color.
        '''
        if check_color(color) == False:
            return grid
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)
        grid_3d[bounding_rectangle & (~selection)] = color
        return grid_3d
    
    def fill_bounding_square_with_color(self, grid, selection, color):
        '''
        Fill the bounding square around the selection with the specified color.
        '''
        if check_color(color) == False:
            return grid
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)
        grid_3d[bounding_square & (~selection)] = color
        return grid_3d