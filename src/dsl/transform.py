import numpy as np
from scipy.ndimage import binary_fill_holes

from dsl.color_select import ColorSelector
from dsl.utilities.checks import check_color
from dsl.utilities.transformation_utilities import (
    create_grid3d,
    find_bounding_rectangle,
    find_bounding_square,
    center_of_mass,
    vectorized_center_of_mass
)


def select_color(grid, method, param):
    """
    Select a color from the grid based on a specified method and parameter.
    """
    colsel = ColorSelector()

    if method == 'color_rank':
        return colsel.rankcolor(grid, param)
    if method == 'shape_rank_nodiag':
        return colsel.rank_largest_shape_color_nodiag(grid, param)
    if method == 'shape_rank_diag':
        return colsel.rank_largest_shape_color_diag(grid, param)


class Transformer:
    """
    A class providing various transformations on a grid. The grid is treated
    as a 3D array for convenience, where the first dimension (depth) can be
    viewed as "layers" or "masks" of the original 2D grid.
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Color transformations
    # -------------------------------------------------------------------------
    def new_color(self, grid, selection, color):
        """
        Change the color of the selected cells in the grid to the specified color.
        If the color already exists in the grid, return the original grid in 3D form.
        """
        grid_3d = create_grid3d(grid, selection)
        if np.sum(grid == color) == 0:
            grid_3d[selection == 1] = color
            return grid_3d
        return np.expand_dims(grid, axis=0)

    def color(self, grid, selection, method, param):
        """
        Apply a color transformation (color_selected) to the selected cells (selection)
        in the grid and return a new 3D grid.
        """
        color_selected = select_color(grid, method, param)
        grid_3d = create_grid3d(grid, selection)
        grid_3d[selection == 1] = color_selected
        return grid_3d

    def fill_with_color(self, grid, selection, method, param):
        """
        Fill all holes inside the single connected shape of the specified color
        and return the modified 3D grid. Holes are identified via binary_fill_holes.
        """
        grid_3d = create_grid3d(grid, selection)
        fill_color = select_color(grid, method, param)
        if not check_color(fill_color):
            return grid_3d

        filled_masks = np.array([binary_fill_holes(i) for i in selection])
        new_masks = filled_masks & (~selection)
        grid_3d[new_masks] = fill_color
        return grid_3d

    def fill_bounding_rectangle_with_color(self, grid, selection, method, param):
        """
        Fill the bounding rectangle around the selection with the specified color.
        """
        color = select_color(grid, method, param)
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)
        grid_3d = np.where(
            (bounding_rectangle & (bounding_rectangle & (1 - selection))) == 1,
            color,
            grid_3d
        )
        return grid_3d

    def fill_bounding_square_with_color(self, grid, selection, method, param):
        """
        Fill the bounding square around the selection with the specified color.
        """
        color = select_color(grid, method, param)
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)
        grid_3d = np.where(
            (bounding_square & (bounding_square & (1 - selection))) == 1,
            color,
            grid_3d
        )
        return grid_3d

    # -------------------------------------------------------------------------
    # Flipping transformations
    # -------------------------------------------------------------------------
    def flipv(self, grid, selection):
        """
        Flip the grid vertically (top-to-bottom) within the bounding rectangle of each selection slice.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)
        flipped_bounding_rectangle = np.flip(bounding_rectangle, axis=1)
        grid_3d[bounding_rectangle] = np.flip(grid_3d, axis=1)[flipped_bounding_rectangle]
        return grid_3d

    def fliph(self, grid, selection):
        """
        Flip the grid horizontally (left-to-right) within the bounding rectangle of each selection slice.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)
        flipped_bounding_rectangle = np.flip(bounding_rectangle, axis=2)
        grid_3d[bounding_rectangle] = np.flip(grid_3d, axis=2)[flipped_bounding_rectangle]
        return grid_3d

    def flip_main_diagonal(self, grid, selection):
        """
        Mirror the selected region along the main diagonal (top-left to bottom-right).
        Only operates on bounding squares; ignores non-square bounding regions.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)

        for i in range(grid_3d.shape[0]):
            mask = bounding_square[i]
            rows, cols = np.where(mask)
            if len(rows) > 0 and len(cols) > 0 and (rows.max() - rows.min() == cols.max() - cols.min()):
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()

                square = grid_3d[i, min_row:max_row + 1, min_col:max_col + 1]
                mirrored = square.T
                grid_3d[i, min_row:max_row + 1, min_col:max_col + 1] = mirrored
        return grid_3d

    def flip_anti_diagonal(self, grid, selection):
        """
        Mirror the selected region along the anti-diagonal (top-right to bottom-left).
        Only operates on bounding squares; ignores non-square bounding regions.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_square = find_bounding_square(selection)

        for i in range(grid_3d.shape[0]):
            mask = bounding_square[i]
            rows, cols = np.where(mask)
            if len(rows) > 0 and len(cols) > 0 and (rows.max() - rows.min() == cols.max() - cols.min()):
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()

                square = grid_3d[i, min_row:max_row + 1, min_col:max_col + 1].copy()
                mirrored = np.flip(np.rot90(square), 1)
                grid_3d[i, min_row:max_row + 1, min_col:max_col + 1] = mirrored
        return grid_3d

    # -------------------------------------------------------------------------
    # Rotation transformations
    # -------------------------------------------------------------------------
    def rotate(self, grid, selection, num_rotations):
        """
        Rotate the selected cells 90 degrees num_rotations times counterclockwise,
        but only if the bounding region is square-shaped.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_masks = find_bounding_square(selection)

        for i in range(bounding_masks.shape[0]):
            bounding_mask = bounding_masks[i]
            rows, cols = np.where(bounding_mask)
            if rows.size == 0 or cols.size == 0:
                continue
            if (rows.max() - rows.min()) == (cols.max() - cols.min()):
                row_start, row_end = rows.min(), rows.max() + 1
                col_start, col_end = cols.min(), cols.max() + 1

                sub_grid = grid_3d[i, row_start:row_end, col_start:col_end]
                rotated_sub_grid = np.rot90(sub_grid, num_rotations)
                grid_3d[i, row_start:row_end, col_start:col_end] = rotated_sub_grid
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

    # -------------------------------------------------------------------------
    # Mirror and duplicate transformations (out-of-grid expansions)
    # -------------------------------------------------------------------------
    def mirror_down(self, grid, selection):
        """
        Mirror the selection vertically below the original grid.
        Works only if rows <= 15. If rows > 15, returns the grid in 3D form.
        """
        d, rows, cols = selection.shape
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
        Mirror the selection vertically above the original grid.
        Works only if rows <= 15. If rows > 15, returns the grid in 3D form.
        """
        d, rows, cols = selection.shape
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
        Mirror the selection horizontally to the right of the original grid.
        Works only if columns <= 15. If columns > 15, returns the grid in 3D form.
        """
        d, rows, cols = selection.shape
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
        Mirror the selection horizontally to the left of the original grid.
        Works only if columns <= 15. If columns > 15, returns the grid in 3D form.
        """
        d, rows, cols = selection.shape
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
        Duplicate the selection horizontally out of the original grid.
        Works only if columns <= 15. Otherwise, returns 3D grid form.
        """
        d, rows, cols = selection.shape
        if cols > 15:
            return create_grid3d(grid, selection)

        grid_3d = create_grid3d(grid, selection)
        new_grid_3d = np.zeros((d, rows, cols * 2))
        new_grid_3d[:, :, :cols] = grid_3d
        new_grid_3d[:, :, cols:][selection.astype(bool)] = grid_3d[selection.astype(bool)]
        return new_grid_3d

    def duplicate_vertically(self, grid, selection):
        """
        Duplicate the selection vertically out of the original grid.
        Works only if rows <= 15. Otherwise, returns 3D grid form.
        """
        d, rows, cols = selection.shape
        if rows > 15:
            return create_grid3d(grid, selection)

        grid_3d = create_grid3d(grid, selection)
        new_grid_3d = np.zeros((d, rows * 2, cols))
        new_grid_3d[:, :rows, :] = grid_3d
        new_grid_3d[:, rows:, :][selection.astype(bool)] = grid_3d[selection.astype(bool)]
        return new_grid_3d

    # -------------------------------------------------------------------------
    # Copy-paste transformations
    # -------------------------------------------------------------------------
    def copy_paste(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops.
        """
        grid_3d = create_grid3d(grid, selection)
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        new_row_idxs = old_row_idxs + shift_y
        new_col_idxs = old_col_idxs + shift_x

        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        layer_idxs = layer_idxs[valid_mask]
        old_row_idxs = old_row_idxs[valid_mask]
        old_col_idxs = old_col_idxs[valid_mask]
        new_row_idxs = new_row_idxs[valid_mask]
        new_col_idxs = new_col_idxs[valid_mask]

        values = grid_3d[layer_idxs, old_row_idxs, old_col_idxs]
        grid_3d[layer_idxs, new_row_idxs, new_col_idxs] = values

        return grid_3d

    def copy_sum(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops,
        summing overlapping values (mod 10).
        """
        grid_3d = create_grid3d(grid, selection)
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        new_row_idxs = old_row_idxs + shift_y
        new_col_idxs = old_col_idxs + shift_x

        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        layer_idxs = layer_idxs[valid_mask]
        old_row_idxs = old_row_idxs[valid_mask]
        old_col_idxs = old_col_idxs[valid_mask]
        new_row_idxs = new_row_idxs[valid_mask]
        new_col_idxs = new_col_idxs[valid_mask]

        values = grid_3d[layer_idxs, old_row_idxs, old_col_idxs]
        np.add.at(grid_3d, (layer_idxs, new_row_idxs, new_col_idxs), values)
        grid_3d = grid_3d % 10

        return grid_3d

    def cut_paste(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops,
        setting the original cells to 0.
        """
        grid_3d = create_grid3d(grid, selection)
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        new_row_idxs = old_row_idxs + shift_y
        new_col_idxs = old_col_idxs + shift_x

        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        values = grid_3d[layer_idxs[valid_mask], old_row_idxs[valid_mask], old_col_idxs[valid_mask]]
        grid_3d[layer_idxs, old_row_idxs, old_col_idxs] = 0
        grid_3d[layer_idxs[valid_mask], new_row_idxs[valid_mask], new_col_idxs[valid_mask]] = values

        return grid_3d

    def cut_sum(self, grid, selection, shift_x, shift_y):
        """
        Shift the selected cells in the grid by (shift_x, shift_y) without using loops,
        summing overlapping values (mod 10), and setting originals to 0.
        """
        grid_3d = create_grid3d(grid, selection)
        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)

        new_row_idxs = old_row_idxs + shift_y
        new_col_idxs = old_col_idxs + shift_x

        valid_mask = (
            (new_row_idxs >= 0) & (new_row_idxs < grid_3d.shape[1]) &
            (new_col_idxs >= 0) & (new_col_idxs < grid_3d.shape[2])
        )

        values = grid_3d[layer_idxs[valid_mask], old_row_idxs[valid_mask], old_col_idxs[valid_mask]]
        grid_3d[layer_idxs, old_row_idxs, old_col_idxs] = 0
        np.add.at(grid_3d, (layer_idxs[valid_mask], new_row_idxs[valid_mask], new_col_idxs[valid_mask]), values)
        grid_3d = grid_3d % 10

        return grid_3d

    def copy_paste_vertically(self, grid, selection):
        """
        For each mask in the selection, copy its selected area and paste it upwards and
        downwards as many times as possible within the grid bounds.
        """
        grid_3d = create_grid3d(grid, selection)
        n_masks, height_of_grid, _ = grid_3d.shape

        rows_with_one = np.any(selection == 1, axis=2)
        first_rows = np.full(n_masks, -1)
        last_rows = np.full(n_masks, -1)

        for idx in range(n_masks):
            row_indices = np.where(rows_with_one[idx])[0]
            if row_indices.size > 0:
                first_rows[idx] = row_indices[0]
                last_rows[idx] = row_indices[-1]

        selection_height = last_rows - first_rows + 1
        factor_up = np.ceil(first_rows / selection_height).astype(int)
        factor_down = np.ceil((height_of_grid - last_rows - 1) / selection_height).astype(int)

        final_transformation = grid_3d.copy()

        for idx in range(n_masks):
            if selection_height[idx] <= 0:
                continue
            grid_layer = final_transformation[idx]
            selection_layer = selection[idx]
            grid_layer_3d = np.expand_dims(grid_layer, axis=0)
            selection_layer_3d = np.expand_dims(selection_layer, axis=0)

            for i in range(factor_up[idx]):
                shift = -(i + 1) * selection_height[idx]
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, 0, shift)

            for i in range(factor_down[idx]):
                shift = (i + 1) * selection_height[idx]
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, 0, shift)

            final_transformation[idx] = grid_layer_3d[0]

        return final_transformation

    def copy_paste_horizontally(self, grid, selection):
        """
        For each mask in the selection, copy its selected area and paste it leftwards and
        rightwards as many times as possible within the grid bounds.
        """
        grid_3d = create_grid3d(grid, selection)
        n_masks, _, width_of_grid = grid_3d.shape

        columns_with_one = np.any(selection == 1, axis=1)
        first_cols = np.full(n_masks, -1)
        last_cols = np.full(n_masks, -1)

        for idx in range(n_masks):
            col_indices = np.where(columns_with_one[idx])[0]
            if col_indices.size > 0:
                first_cols[idx] = col_indices[0]
                last_cols[idx] = col_indices[-1]

        selection_width = last_cols - first_cols + 1
        factor_left = np.ceil(first_cols / selection_width).astype(int)
        factor_right = np.ceil((width_of_grid - last_cols - 1) / selection_width).astype(int)

        final_transformation = grid_3d.copy()

        for idx in range(n_masks):
            if selection_width[idx] <= 0:
                continue
            grid_layer = final_transformation[idx]
            selection_layer = selection[idx]
            grid_layer_3d = np.expand_dims(grid_layer, axis=0)
            selection_layer_3d = np.expand_dims(selection_layer, axis=0)

            for i in range(factor_left[idx]):
                shift = -(i + 1) * selection_width[idx]
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, shift, 0)

            for i in range(factor_right[idx]):
                shift = (i + 1) * selection_width[idx]
                grid_layer_3d = self.copy_paste(grid_layer_3d, selection_layer_3d, shift, 0)

            final_transformation[idx] = grid_layer_3d[0]

        return final_transformation

    # -------------------------------------------------------------------------
    # Gravitate transformations (whole selection moves together)
    # -------------------------------------------------------------------------
    def gravitate_whole_downwards_paste(self, grid, selection):
        """
        Copy and paste the selected cells in the grid downwards as a whole
        until they reach the bottom of the grid or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        row_indices = np.arange(rows).reshape(1, rows, 1)
        sel_row_indices = np.where(selection, row_indices, -1)
        max_row_sel = sel_row_indices.max(axis=1)
        selection_exists_in_column = (max_row_sel != -1)
        max_row_sel_expanded = max_row_sel[:, None, :]

        mask_below_selection = row_indices > max_row_sel_expanded
        obstacles_below = (grid_without_selection != 0) & mask_below_selection
        obstacle_positions = np.where(obstacles_below, row_indices, rows)
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, rows + 1)

        shift_per_column = obstacle_positions.min(axis=1) - max_row_sel - 1
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)
        shift_per_depth = np.min(shift_per_column, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, rows)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_y = shift_per_depth[layer_idxs]
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)
        return grid_3d

    def gravitate_whole_upwards_paste(self, grid, selection):
        """
        Copy and paste the selected cells in the grid upwards as a whole
        until they reach the top of the grid or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        row_indices = np.arange(rows).reshape(1, rows, 1)
        sel_row_indices = np.where(selection, row_indices, rows)
        min_row_sel = sel_row_indices.min(axis=1)
        selection_exists_in_column = (min_row_sel != rows)
        min_row_sel_expanded = min_row_sel[:, None, :]

        mask_above_selection = row_indices < min_row_sel_expanded
        obstacles_above = (grid_without_selection != 0) & mask_above_selection
        obstacle_positions = np.where(obstacles_above, row_indices, -1)
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, -1)

        shift_per_column = min_row_sel - obstacle_positions.max(axis=1) - 1
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)
        shift_per_depth = np.min(shift_per_column, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, rows).astype(int)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_y = -shift_per_depth[layer_idxs]
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)
        return grid_3d

    def gravitate_whole_right_paste(self, grid, selection):
        """
        Copy and paste the selected cells in the grid to the right as a whole
        until they reach the right edge of the grid or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        col_indices = np.arange(cols).reshape(1, 1, cols)
        sel_col_indices = np.where(selection, col_indices, -1)
        max_col_sel = sel_col_indices.max(axis=2)
        selection_exists_in_row = (max_col_sel != -1)
        max_col_sel_expanded = max_col_sel[:, :, None]

        mask_right_selection = col_indices > max_col_sel_expanded
        obstacles_right = (grid_without_selection != 0) & mask_right_selection
        obstacle_positions = np.where(obstacles_right, col_indices, cols)
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, cols + 1)

        shift_per_row = obstacle_positions.min(axis=2) - max_col_sel - 1
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)
        shift_per_depth = np.min(shift_per_row, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_x = shift_per_depth[layer_idxs]
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)
        return grid_3d

    def gravitate_whole_left_paste(self, grid, selection):
        """
        Copy and paste the selected cells in the grid to the left as a whole
        until they reach the left edge of the grid or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        col_indices = np.arange(cols).reshape(1, 1, cols)
        sel_col_indices = np.where(selection, col_indices, cols)
        min_col_sel = sel_col_indices.min(axis=2)
        selection_exists_in_row = (min_col_sel != cols)
        min_col_sel_expanded = min_col_sel[:, :, None]

        mask_left_selection = col_indices < min_col_sel_expanded
        obstacles_left = (grid_without_selection != 0) & mask_left_selection
        obstacle_positions = np.where(obstacles_left, col_indices, -1)
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, -1)

        shift_per_row = min_col_sel - obstacle_positions.max(axis=2) - 1
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)
        shift_per_depth = np.min(shift_per_row, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_x = -shift_per_depth[layer_idxs]
        grid_3d = self.copy_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)
        return grid_3d

    def gravitate_whole_downwards_cut(self, grid, selection):
        """
        Shift the selected cells in the grid downwards as a whole (cut-paste)
        until they reach the bottom or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        row_indices = np.arange(rows).reshape(1, rows, 1)
        sel_row_indices = np.where(selection, row_indices, -1)
        max_row_sel = sel_row_indices.max(axis=1)
        selection_exists_in_column = (max_row_sel != -1)
        max_row_sel_expanded = max_row_sel[:, None, :]

        mask_below_selection = row_indices > max_row_sel_expanded
        obstacles_below = (grid_without_selection != 0) & mask_below_selection
        obstacle_positions = np.where(obstacles_below, row_indices, rows)
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, rows + 1)

        shift_per_column = obstacle_positions.min(axis=1) - max_row_sel - 1
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)
        shift_per_depth = np.min(shift_per_column, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, rows)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_y = shift_per_depth[layer_idxs]
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)
        return grid_3d

    def gravitate_whole_upwards_cut(self, grid, selection):
        """
        Shift the selected cells in the grid upwards as a whole (cut-paste)
        until they reach the top or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        row_indices = np.arange(rows).reshape(1, rows, 1)
        sel_row_indices = np.where(selection, row_indices, rows)
        min_row_sel = sel_row_indices.min(axis=1)
        selection_exists_in_column = (min_row_sel != rows)
        min_row_sel_expanded = min_row_sel[:, None, :]

        mask_above_selection = row_indices < min_row_sel_expanded
        obstacles_above = (grid_without_selection != 0) & mask_above_selection
        obstacle_positions = np.where(obstacles_above, row_indices, -1)
        obstacle_positions = np.where(selection_exists_in_column[:, None, :], obstacle_positions, -1)

        shift_per_column = min_row_sel - obstacle_positions.max(axis=1) - 1
        shift_per_column = np.where(selection_exists_in_column, shift_per_column, rows + 1)
        shift_per_depth = np.min(shift_per_column, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, rows).astype(int)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_y = -shift_per_depth[layer_idxs]
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=0, shift_y=shift_y)
        return grid_3d

    def gravitate_whole_right_cut(self, grid, selection):
        """
        Shift the selected cells in the grid to the right as a whole (cut-paste)
        until they reach the right edge or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        col_indices = np.arange(cols).reshape(1, 1, cols)
        sel_col_indices = np.where(selection, col_indices, -1)
        max_col_sel = sel_col_indices.max(axis=2)
        selection_exists_in_row = (max_col_sel != -1)
        max_col_sel_expanded = max_col_sel[:, :, None]

        mask_right_selection = col_indices > max_col_sel_expanded
        obstacles_right = (grid_without_selection != 0) & mask_right_selection
        obstacle_positions = np.where(obstacles_right, col_indices, cols)
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, cols + 1)

        shift_per_row = obstacle_positions.min(axis=2) - max_col_sel - 1
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)
        shift_per_depth = np.min(shift_per_row, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_x = shift_per_depth[layer_idxs]
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)
        return grid_3d

    def gravitate_whole_left_cut(self, grid, selection):
        """
        Shift the selected cells in the grid to the left as a whole (cut-paste)
        until they reach the left edge or collide with non-zero cells.
        """
        grid_3d = create_grid3d(grid, selection)
        depth, rows, cols = selection.shape
        grid_without_selection = grid_3d.copy()
        indices = np.nonzero(selection)
        grid_without_selection[indices] = 0

        col_indices = np.arange(cols).reshape(1, 1, cols)
        sel_col_indices = np.where(selection, col_indices, cols)
        min_col_sel = sel_col_indices.min(axis=2)
        selection_exists_in_row = (min_col_sel != cols)
        min_col_sel_expanded = min_col_sel[:, :, None]

        mask_left_selection = col_indices < min_col_sel_expanded
        obstacles_left = (grid_without_selection != 0) & mask_left_selection
        obstacle_positions = np.where(obstacles_left, col_indices, -1)
        obstacle_positions = np.where(selection_exists_in_row[:, :, None], obstacle_positions, -1)

        shift_per_row = min_col_sel - obstacle_positions.max(axis=2) - 1
        shift_per_row = np.where(selection_exists_in_row, shift_per_row, cols + 1)
        shift_per_depth = np.min(shift_per_row, axis=1)
        shift_per_depth = np.clip(shift_per_depth, 0, cols).astype(int)

        layer_idxs, old_row_idxs, old_col_idxs = np.where(selection)
        shift_x = -shift_per_depth[layer_idxs]
        grid_3d = self.cut_paste(grid_3d, selection, shift_x=shift_x, shift_y=0)
        return grid_3d

    # -------------------------------------------------------------------------
    # Individual-cell gravity transformations (non-vectorized per-tile moves)
    # -------------------------------------------------------------------------
    def down_gravity(self, grid, selection):
        """
        Move the selected cells downwards individually until they hit non-zero cells
        or the bottom of the grid.
        """
        grid_3d = create_grid3d(grid, selection)
        num_layers, num_rows, num_cols = grid_3d.shape

        for layer_idx in range(num_layers):
            selection_layer = selection[layer_idx]
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells

            for i in range(len(selected_rows) - 1, -1, -1):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                for target_row in range(row + 1, num_rows):
                    if grid_3d[layer_idx, target_row, col] != 0:
                        grid_3d[layer_idx, target_row - 1, col] = value
                        selection_layer[target_row - 1, col] = 1
                        break
                else:
                    grid_3d[layer_idx, num_rows - 1, col] = value
                    selection_layer[num_rows - 1, col] = 1
        return grid_3d

    def up_gravity(self, grid, selection):
        """
        Move the selected cells upwards individually until they hit non-zero cells
        or the top of the grid.
        """
        grid_3d = create_grid3d(grid, selection)
        num_layers, num_rows, num_cols = grid_3d.shape

        for layer_idx in range(num_layers):
            selection_layer = selection[layer_idx]
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells

            for i in range(len(selected_rows)):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                for target_row in range(row - 1, -1, -1):
                    if grid_3d[layer_idx, target_row, col] != 0:
                        grid_3d[layer_idx, target_row + 1, col] = value
                        selection_layer[target_row + 1, col] = 1
                        break
                else:
                    grid_3d[layer_idx, 0, col] = value
                    selection_layer[0, col] = 1
        return grid_3d

    def right_gravity(self, grid, selection):
        """
        Move the selected cells rightwards individually until they hit non-zero cells
        or the rightmost column.
        """
        grid_3d = create_grid3d(grid, selection)
        num_layers, num_rows, num_cols = grid_3d.shape

        for layer_idx in range(num_layers):
            selection_layer = selection[layer_idx]
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells

            for i in range(len(selected_cols) - 1, -1, -1):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                for target_col in range(col + 1, num_cols):
                    if grid_3d[layer_idx, row, target_col] != 0:
                        grid_3d[layer_idx, row, target_col - 1] = value
                        selection_layer[row, target_col - 1] = 1
                        break
                else:
                    grid_3d[layer_idx, row, num_cols - 1] = value
                    selection_layer[row, num_cols - 1] = 1
        return grid_3d

    def left_gravity(self, grid, selection):
        """
        Move the selected cells leftwards individually until they hit non-zero cells
        or the leftmost column.
        """
        grid_3d = create_grid3d(grid, selection)
        num_layers, num_rows, num_cols = grid_3d.shape

        for layer_idx in range(num_layers):
            selection_layer = selection[layer_idx]
            selected_cells = np.where(selection_layer == 1)
            selected_rows, selected_cols = selected_cells

            for i in range(len(selected_cols) - 1, -1, -1):
                row, col = selected_rows[i], selected_cols[i]
                value = grid_3d[layer_idx, row, col]
                grid_3d[layer_idx, row, col] = 0
                selection_layer[row, col] = 0

                for target_col in range(col - 1, -1, -1):
                    if grid_3d[layer_idx, row, target_col] != 0:
                        grid_3d[layer_idx, row, target_col + 1] = value
                        selection_layer[row, target_col + 1] = 1
                        break
                else:
                    grid_3d[layer_idx, row, 0] = value
                    selection_layer[row, 0] = 1
        return grid_3d

    # -------------------------------------------------------------------------
    # Upscale transformations
    # -------------------------------------------------------------------------
    def vupscale(self, grid, selection, scale_factor):
        """
        Vertically upscale the selection in the grid by a specified scale factor,
        and cap the upscaled selection to match the original size.
        """
        selection_3d_grid = create_grid3d(grid, selection)
        depth, original_rows, original_cols = selection.shape

        upscaled_selection = np.repeat(selection, scale_factor, axis=1)
        upscaled_selection_3d_grid = np.repeat(selection_3d_grid, scale_factor, axis=1)

        if original_rows % 2 == 0:
            half_rows_top, half_rows_bottom = original_rows // 2, original_rows // 2
        else:
            half_rows_top, half_rows_bottom = original_rows // 2 + 1, original_rows // 2

        capped_selection = np.zeros((depth, original_rows, original_cols), dtype=bool)
        capped_upscaled_grid = np.zeros((depth, original_rows, original_cols))

        for layer_idx in range(depth):
            original_com = center_of_mass(selection[layer_idx])[0]
            upscaled_com = center_of_mass(upscaled_selection[layer_idx])[0]

            lower_bound = min(int(upscaled_com + half_rows_bottom), original_rows * scale_factor)
            upper_bound = max(int(upscaled_com - half_rows_top), 0)

            if lower_bound >= original_rows * scale_factor:
                lower_bound = original_rows * scale_factor
                upper_bound = lower_bound - original_rows
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_rows

            capped_selection[layer_idx] = upscaled_selection[layer_idx, upper_bound:lower_bound, :]
            capped_com = center_of_mass(capped_selection[layer_idx])[0]
            offset = capped_com - original_com
            lower_bound += offset
            upper_bound += offset

            if lower_bound >= original_rows * scale_factor:
                lower_bound = original_rows * scale_factor
                upper_bound = lower_bound - original_rows
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_rows

            capped_selection[layer_idx] = upscaled_selection[layer_idx, upper_bound:lower_bound, :]
            capped_upscaled_grid[layer_idx] = upscaled_selection_3d_grid[layer_idx, upper_bound:lower_bound, :]

        selection_3d_grid[selection == 1] = 0
        selection_3d_grid[capped_selection] = capped_upscaled_grid[capped_selection].ravel()
        return selection_3d_grid

    def hupscale(self, grid, selection, scale_factor):
        """
        Horizontally upscale the selection in the grid by a specified scale factor,
        and cap the upscaled selection to match the original size.
        """
        selection_3d_grid = create_grid3d(grid, selection)
        depth, original_rows, original_cols = selection.shape

        upscaled_selection = np.repeat(selection, scale_factor, axis=2)
        upscaled_selection_3d_grid = np.repeat(selection_3d_grid, scale_factor, axis=2)
        upscaled_cols = upscaled_selection.shape[2]

        if original_cols % 2 == 0:
            half_cols_left, half_cols_right = original_cols // 2, original_cols // 2
        else:
            half_cols_left, half_cols_right = original_cols // 2 + 1, original_cols // 2

        capped_selection = np.zeros((depth, original_rows, original_cols), dtype=bool)
        capped_upscaled_grid = np.zeros((depth, original_rows, original_cols))

        for layer_idx in range(depth):
            original_com = center_of_mass(selection[layer_idx])[1]
            upscaled_com = center_of_mass(upscaled_selection[layer_idx])[1]

            lower_bound = min(int(upscaled_com + half_cols_right), upscaled_cols)
            upper_bound = max(int(upscaled_com - half_cols_left), 0)

            if lower_bound >= upscaled_cols:
                lower_bound = upscaled_cols
                upper_bound = lower_bound - original_cols
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_cols

            capped_selection[layer_idx] = upscaled_selection[layer_idx, :, upper_bound:lower_bound]
            capped_com = center_of_mass(capped_selection[layer_idx])[1]
            offset = int(capped_com - original_com)
            lower_bound += offset
            upper_bound += offset

            if lower_bound >= upscaled_cols:
                lower_bound = upscaled_cols
                upper_bound = lower_bound - original_cols
            elif upper_bound <= 0:
                upper_bound = 0
                lower_bound = upper_bound + original_cols

            capped_selection[layer_idx] = upscaled_selection[layer_idx, :, upper_bound:lower_bound]
            capped_upscaled_grid[layer_idx] = upscaled_selection_3d_grid[layer_idx, :, upper_bound:lower_bound]

        selection_3d_grid[selection == 1] = 0
        capped_mask = capped_selection.astype(bool)
        selection_3d_grid[capped_mask] = capped_upscaled_grid[capped_mask].ravel()
        return selection_3d_grid

    # -------------------------------------------------------------------------
    # Delete / crop transformations
    # -------------------------------------------------------------------------
    def crop(self, grid, selection):
        """
        Crop the grid to the bounding rectangle around the selection.
        Use -1 as the value for cells outside the selection.
        """
        grid_3d = create_grid3d(grid, selection)
        bounding_rectangle = find_bounding_rectangle(selection)

        for i in range(selection.shape[0]):
            if not bounding_rectangle[i].any():
                bounding_rectangle[i] = np.ones_like(grid_3d[i], dtype=bool)

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
        """
        Change the background color of the grid to the specified color.
        The background color is determined by the most common color in the grid.
        """
        grid3d = create_grid3d(grid, selection)
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid)
        background_color = int(background_color)
        grid3d[grid3d == background_color] = new_color

        if not check_color(new_color):
            return grid3d
        return grid3d

    def change_selection_to_background_color(self, grid, selection):
        """
        Change the selected cells in the grid to the background color,
        where the background color is determined by the most common color.
        """
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid)
        grid_3d = create_grid3d(grid, selection)
        grid_3d[selection == 1] = background_color
        return grid_3d

    def vectorized_vupscale(self, grid, selection, scale_factor):
        """
        Upscale the selection in the grid vertically by a specified scale factor,
        then overwrite existing values (vectorized approach).
        """
        grid_3d = create_grid3d(grid, selection)
        depth, original_rows, original_cols = selection.shape

        upscaled_selection = np.repeat(selection, scale_factor, axis=1)
        upscaled_grid = np.repeat(grid_3d * selection, scale_factor, axis=1)
        upscaled_rows = upscaled_selection.shape[1]

        com_original = vectorized_center_of_mass(selection)
        com_upscaled = vectorized_center_of_mass(upscaled_selection)

        shift = (com_original - com_upscaled)
        shifted_row_indices = (
            np.arange(upscaled_rows).reshape(1, upscaled_rows, 1) + shift
        )
        shifted_row_indices = np.broadcast_to(shifted_row_indices, upscaled_selection.shape)

        valid_mask = (
            (shifted_row_indices >= 0) &
            (shifted_row_indices < original_rows) &
            upscaled_selection
        )

        indices = np.argwhere(valid_mask)
        d = indices[:, 0]
        r_up = indices[:, 1]
        c = indices[:, 2]
        shifted_rows = shifted_row_indices[valid_mask].flatten()
        values = upscaled_grid[valid_mask].flatten()

        rev_indices = np.arange(len(d) - 1, -1, -1)
        d_rev = d[rev_indices]
        shifted_rows_rev = shifted_rows[rev_indices]
        c_rev = c[rev_indices]
        values_rev = values[rev_indices]

        final_grid = np.zeros((depth, original_rows, original_cols), dtype=grid_3d.dtype)
        final_grid[d_rev, shifted_rows_rev, c_rev] = values_rev

        grid_3d[selection] = 0
        grid_3d[final_grid != 0] = final_grid[final_grid != 0]
        return grid_3d
