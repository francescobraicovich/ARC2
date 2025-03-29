# #%%
# from numba.experimental import jitclass
# import numba as nb
# import numpy as np
# import time

# # =============================================================================
# # Utility functions (numba-compatible versions)
# # =============================================================================

# @nb.njit
# def binary_fill_holes_numba(mask):
#     """
#     A very simple (and not full) numba implementation to fill holes in a binary mask.
#     (For real applications, a more robust implementation would be needed.)
#     """
#     rows, cols = mask.shape
#     out = mask.copy()
#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             if not mask[i, j]:
#                 # if all four neighbors are True, then fill the hole
#                 if mask[i-1, j] and mask[i+1, j] and mask[i, j-1] and mask[i, j+1]:
#                     out[i, j] = True
#     return out

# @nb.njit
# def create_grid3d_numba(grid, selection):
#     """
#     Create a 3D grid with one layer equal to grid.
#     (In our DSL, the grid is “expanded” along the first dimension.)
#     """
#     n, m = grid.shape
#     grid3d = np.empty((1, n, m), dtype=grid.dtype)
#     for i in range(n):
#         for j in range(m):
#             grid3d[0, i, j] = grid[i, j]
#     return grid3d

# @nb.njit
# def find_bounding_rectangle_numba(selection):
#     """
#     Find the bounding rectangle of a 2D boolean mask.
#     Returns a boolean mask (of same shape) that is True exactly within the bounding box.
#     """
#     rows, cols = selection.shape
#     min_row = rows
#     max_row = -1
#     min_col = cols
#     max_col = -1
#     for i in range(rows):
#         for j in range(cols):
#             if selection[i, j]:
#                 if i < min_row:
#                     min_row = i
#                 if i > max_row:
#                     max_row = i
#                 if j < min_col:
#                     min_col = j
#                 if j > max_col:
#                     max_col = j
#     out = np.zeros((rows, cols), dtype=np.bool_)
#     if max_row >= min_row and max_col >= min_col:
#         for i in range(min_row, max_row+1):
#             for j in range(min_col, max_col+1):
#                 out[i, j] = True
#     return out

# @nb.njit
# def find_bounding_square_numba(selection):
#     """
#     For simplicity, return the bounding rectangle (ideally a square).
#     In a full implementation, one might force a square by padding.
#     """
#     return find_bounding_rectangle_numba(selection)

# @nb.njit
# def center_of_mass_numba(mask):
#     """
#     Compute the center of mass of a boolean mask.
#     """
#     rows, cols = mask.shape
#     total = 0.0
#     sum_i = 0.0
#     sum_j = 0.0
#     for i in range(rows):
#         for j in range(cols):
#             if mask[i, j]:
#                 total += 1.0
#                 sum_i += i
#                 sum_j += j
#     if total == 0:
#         return (rows // 2, cols // 2)
#     return (sum_i / total, sum_j / total)

# @nb.njit
# def vectorized_center_of_mass_numba(selection):
#     """
#     Compute the center of mass for each layer in a 3D selection.
#     """
#     depth = selection.shape[0]
#     centers = np.empty((depth, 2), dtype=np.float64)
#     for d in range(depth):
#         centers[d, 0], centers[d, 1] = center_of_mass_numba(selection[d])
#     return centers

# # -----------------------------------------------------------------------------
# # Simple color selection functions (numba versions)
# # -----------------------------------------------------------------------------

# @nb.njit
# def rankcolor_numba(grid, rank):
#     """
#     Count the frequency of each color (assumed to be in 0..9) and return the color with the given rank.
#     (A simple bubble-sort is used on the fixed-length array.)
#     """
#     flat = grid.ravel()
#     counts = np.zeros(10, dtype=np.int64)
#     for i in range(flat.size):
#         counts[flat[i]] += 1
#     colors = np.arange(10)
#     for i in range(10):
#         for j in range(10 - i - 1):
#             if counts[j] < counts[j+1]:
#                 temp = counts[j]
#                 counts[j] = counts[j+1]
#                 counts[j+1] = temp
#                 temp = colors[j]
#                 colors[j] = colors[j+1]
#                 colors[j+1] = temp
#     if rank < 10:
#         return colors[rank]
#     else:
#         return colors[9]

# @nb.njit
# def rank_largest_shape_color_nodiag_numba(grid, rank):
#     # For demonstration, simply call rankcolor_numba.
#     return rankcolor_numba(grid, rank)

# @nb.njit
# def rank_largest_shape_color_diag_numba(grid, rank):
#     # For demonstration, simply call rankcolor_numba.
#     return rankcolor_numba(grid, rank)

# @nb.njit
# def select_color_numba(grid, method_code, param):
#     """
#     Select a color based on the given method_code and parameter.
#     Use:
#       0 for 'color_rank'
#       1 for 'shape_rank_nodiag'
#       2 for 'shape_rank_diag'
#     """
#     if method_code == 0:
#         return rankcolor_numba(grid, param)
#     elif method_code == 1:
#         return rank_largest_shape_color_nodiag_numba(grid, param)
#     elif method_code == 2:
#         return rank_largest_shape_color_diag_numba(grid, param)
#     else:
#         return -1

# # =============================================================================
# # Transformer class (using jitclass)
# # =============================================================================

# spec = {}  # No attributes needed
# @nb.experimental.jitclass(spec)
# class TransformerNumba:
#     def __init__(self):
#         pass

#     # --- Color Transformations ---
#     def new_color(self, grid, selection, color):
#         """
#         Change the color of the selected cells to the specified color.
#         If the color already exists in grid, return grid as 3D.
#         """
#         grid_3d = create_grid3d_numba(grid, selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         total = 0
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if grid[i, j] == color:
#                     total += 1
#         if total == 0:
#             for i in range(n_rows):
#                 for j in range(n_cols):
#                     if selection[i, j]:
#                         grid_3d[0, i, j] = color
#             return grid_3d
#         else:
#             out = np.empty((1, n_rows, n_cols), dtype=grid.dtype)
#             for i in range(n_rows):
#                 for j in range(n_cols):
#                     out[0, i, j] = grid[i, j]
#             return out

#     def color(self, grid, selection, method_code, param):
#         """
#         Apply a color transformation to the selected cells.
#         """
#         c = select_color_numba(grid, method_code, param)
#         grid_3d = create_grid3d_numba(grid, selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if selection[i, j]:
#                     grid_3d[0, i, j] = c
#         return grid_3d

#     def fill_with_color(self, grid, selection, method_code, param):
#         """
#         Fill holes within the connected shape by applying binary_fill_holes.
#         """
#         grid_3d = create_grid3d_numba(grid, selection)
#         fill_color = select_color_numba(grid, method_code, param)
#         if not (fill_color >= 0 and fill_color < 10):
#             return grid_3d
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         filled = binary_fill_holes_numba(selection)
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if filled[i, j] and (not selection[i, j]):
#                     grid_3d[0, i, j] = fill_color
#         return grid_3d

#     def fill_bounding_rectangle_with_color(self, grid, selection, method_code, param):
#         """
#         Fill the bounding rectangle around the selection with a specified color.
#         """
#         color = select_color_numba(grid, method_code, param)
#         grid_3d = create_grid3d_numba(grid, selection)
#         bound = find_bounding_rectangle_numba(selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if bound[i, j] and (not selection[i, j]):
#                     grid_3d[0, i, j] = color
#         return grid_3d

#     # --- Flipping Transformations ---
#     def flipv(self, grid, selection):
#         """
#         Flip the grid vertically within the bounding rectangle.
#         """
#         grid_3d = create_grid3d_numba(grid, selection)
#         bound = find_bounding_rectangle_numba(selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         flipped_grid = np.empty_like(grid_3d)
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 flipped_grid[0, i, j] = grid_3d[0, n_rows - 1 - i, j]
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if bound[i, j]:
#                     grid_3d[0, i, j] = flipped_grid[0, i, j]
#         return grid_3d

#     # --- Rotation Transformations ---
#     def rotate(self, grid, selection, num_rotations):
#         """
#         Rotate the selected area (assumed square) by 90° * num_rotations.
#         """
#         grid_3d = create_grid3d_numba(grid, selection)
#         bound = find_bounding_square_numba(selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         min_row = n_rows
#         max_row = -1
#         min_col = n_cols
#         max_col = -1
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if bound[i, j]:
#                     if i < min_row:
#                         min_row = i
#                     if i > max_row:
#                         max_row = i
#                     if j < min_col:
#                         min_col = j
#                     if j > max_col:
#                         max_col = j
#         if max_row >= min_row and max_col >= min_col and ((max_row - min_row) == (max_col - min_col)):
#             size = max_row - min_row + 1
#             sub_grid = np.empty((size, size), dtype=grid.dtype)
#             for i in range(size):
#                 for j in range(size):
#                     sub_grid[i, j] = grid_3d[0, min_row + i, min_col + j]
#             for _ in range(num_rotations):
#                 temp = np.empty_like(sub_grid)
#                 for i in range(size):
#                     for j in range(size):
#                         temp[i, j] = sub_grid[size - 1 - j, i]
#                 sub_grid = temp
#             for i in range(size):
#                 for j in range(size):
#                     grid_3d[0, min_row + i, min_col + j] = sub_grid[i, j]
#         return grid_3d

#     def rotate_90(self, grid, selection):
#         return self.rotate(grid, selection, 1)

#     def rotate_180(self, grid, selection):
#         return self.rotate(grid, selection, 2)

#     def rotate_270(self, grid, selection):
#         return self.rotate(grid, selection, 3)

#     # --- Crop and Delete ---
#     def crop(self, grid, selection):
#         """
#         Crop the grid to the bounding rectangle; cells outside are set to -1.
#         """
#         grid_3d = create_grid3d_numba(grid, selection)
#         bound = find_bounding_rectangle_numba(selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if not bound[i, j]:
#                     grid_3d[0, i, j] = -1
#         return grid_3d

#     def delete(self, grid, selection):
#         """
#         Set the selected cells to 0.
#         """
#         grid_3d = create_grid3d_numba(grid, selection)
#         n_rows = grid.shape[0]
#         n_cols = grid.shape[1]
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if selection[i, j]:
#                     grid_3d[0, i, j] = 0
#         return grid_3d

# # =============================================================================
# # Final Test Block and Running Time Tests
# # =============================================================================

# def main():
#     # Create a 30x30 random grid with colors 0..9 and a simple selection (cells equal to 3)
#     grid = np.random.randint(0, 10, size=(30, 30)).astype(np.int64)
#     selection = (grid == 3)
    
#     transformer = TransformerNumba()
    
#     result_new_color = transformer.new_color(grid, selection, 5)
#     result_flipv = transformer.flipv(grid, selection)
#     result_rotate90 = transformer.rotate_90(grid, selection)
#     result_crop = transformer.crop(grid, selection)
    
#     print("new_color result shape:", result_new_color.shape)
#     print("flipv result shape:", result_flipv.shape)
#     print("rotate_90 result shape:", result_rotate90.shape)
#     print("crop result shape:", result_crop.shape)

# def run_time_tests():
#     # Create a large random grid (1000x1000) with colors 0..9 and a selection mask (cells equal to 3)
#     grid = np.random.randint(0, 10, size=(1000, 1000)).astype(np.int64)
#     selection = (grid == 3)
    
#     transformer = TransformerNumba()
    
#     # Warm-up: call each method once to trigger JIT compilation.
#     _ = transformer.new_color(grid, selection, 5)
#     _ = transformer.color(grid, selection, 0, 2)  # 0 stands for 'color_rank'
#     _ = transformer.fill_with_color(grid, selection, 0, 4)
#     _ = transformer.fill_bounding_rectangle_with_color(grid, selection, 0, 3)
#     _ = transformer.flipv(grid, selection)
#     _ = transformer.rotate_90(grid, selection)
#     _ = transformer.crop(grid, selection)
#     _ = transformer.delete(grid, selection)
    
#     iterations = 100

#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.new_color(grid, selection, 5)
#     end = time.time()
#     print("TransformerNumba.new_color average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.color(grid, selection, 0, 2)
#     end = time.time()
#     print("TransformerNumba.color average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.fill_with_color(grid, selection, 0, 4)
#     end = time.time()
#     print("TransformerNumba.fill_with_color average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.fill_bounding_rectangle_with_color(grid, selection, 0, 3)
#     end = time.time()
#     print("TransformerNumba.fill_bounding_rectangle_with_color average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.flipv(grid, selection)
#     end = time.time()
#     print("TransformerNumba.flipv average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.rotate_90(grid, selection)
#     end = time.time()
#     print("TransformerNumba.rotate_90 average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.crop(grid, selection)
#     end = time.time()
#     print("TransformerNumba.crop average time: {:.6f} s".format((end - start) / iterations))
    
#     start = time.time()
#     for _ in range(iterations):
#         _ = transformer.delete(grid, selection)
#     end = time.time()
#     print("TransformerNumba.delete average time: {:.6f} s".format((end - start) / iterations))

# if __name__ == '__main__':
#     # Run the functional tests and then the timing tests.
#     main()
#     run_time_tests()

























#%%
from numba.experimental import jitclass
import numba as nb
import numpy as np
import time

# =============================================================================
# Utility functions (numba-compatible versions) – operate on 2D arrays.
# =============================================================================

@nb.njit(parallel=True)
def binary_fill_holes_numba(mask):
    """
    A simple numba implementation to fill holes in a 2D boolean mask.
    (For robust applications, a full hole-filling algorithm is required.)
    """
    rows, cols = mask.shape
    out = mask.copy()
    for i in nb.prange(1, rows-1):
        for j in range(1, cols-1):
            if not mask[i, j]:
                if mask[i-1, j] and mask[i+1, j] and mask[i, j-1] and mask[i, j+1]:
                    out[i, j] = True
    return out

@nb.njit(parallel=True)
def create_grid3d_numba(grid, selection):
    """
    Create a 3D grid with one layer equal to grid.
    (Here grid is 2D and selection is a 2D mask.)
    """
    n, m = grid.shape
    grid3d = np.empty((1, n, m), dtype=grid.dtype)
    for i in nb.prange(n):
        for j in range(m):
            grid3d[0, i, j] = grid[i, j]
    return grid3d

@nb.njit
def find_bounding_rectangle_numba(selection):
    """
    Compute a 2D boolean mask that is True exactly in the bounding rectangle of the 2D selection.
    """
    rows, cols = selection.shape
    min_row = rows
    max_row = -1
    min_col = cols
    max_col = -1
    for i in range(rows):
        for j in range(cols):
            if selection[i, j]:
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    out = np.zeros((rows, cols), dtype=np.bool_)
    if max_row >= min_row and max_col >= min_col:
        for i in range(min_row, max_row+1):
            for j in range(min_col, max_col+1):
                out[i, j] = True
    return out

@nb.njit
def find_bounding_square_numba(selection):
    """
    For simplicity, return the bounding rectangle.
    """
    return find_bounding_rectangle_numba(selection)

@nb.njit
def center_of_mass_numba(mask):
    """
    Compute the center of mass of a 2D boolean mask.
    """
    rows, cols = mask.shape
    total = 0.0
    sum_i = 0.0
    sum_j = 0.0
    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:
                total += 1.0
                sum_i += i
                sum_j += j
    if total == 0:
        return (rows // 2, cols // 2)
    return (sum_i / total, sum_j / total)

@nb.njit
def vectorized_center_of_mass_numba(selection):
    """
    For compatibility – this version computes the center of mass for a 2D mask.
    """
    return center_of_mass_numba(selection)

# -----------------------------------------------------------------------------
# Simple color selection functions (numba versions)
# -----------------------------------------------------------------------------

@nb.njit
def rankcolor_numba(grid, rank):
    """
    Count frequencies of colors (assumed 0..9) in grid and return the color of the given rank.
    """
    flat = grid.ravel()
    counts = np.zeros(10, dtype=np.int64)
    for i in range(flat.size):
        counts[flat[i]] += 1
    colors = np.arange(10)
    # Bubble-sort on fixed-length arrays.
    for i in range(10):
        for j in range(10 - i - 1):
            if counts[j] < counts[j+1]:
                temp = counts[j]
                counts[j] = counts[j+1]
                counts[j+1] = temp
                temp = colors[j]
                colors[j] = colors[j+1]
                colors[j+1] = temp
    if rank < 10:
        return colors[rank]
    else:
        return colors[9]

@nb.njit
def rank_largest_shape_color_nodiag_numba(grid, rank):
    return rankcolor_numba(grid, rank)

@nb.njit
def rank_largest_shape_color_diag_numba(grid, rank):
    return rankcolor_numba(grid, rank)

@nb.njit
def select_color_numba(grid, method_code, param):
    """
    Use method_code:
      0: 'color_rank'
      1: 'shape_rank_nodiag'
      2: 'shape_rank_diag'
    """
    if method_code == 0:
        return rankcolor_numba(grid, param)
    elif method_code == 1:
        return rank_largest_shape_color_nodiag_numba(grid, param)
    elif method_code == 2:
        return rank_largest_shape_color_diag_numba(grid, param)
    else:
        return -1

# =============================================================================
# Transformer class (using jitclass)
# =============================================================================

spec = {}  # No attributes needed
@nb.experimental.jitclass(spec)
class TransformerNumba:
    def __init__(self):
        pass

    # --- Color Transformations ---
    def new_color(self, grid, selection, color):
        """
        Change the color of the selected cells to the specified color.
        If the color already exists in grid, return the original grid in 3D form.
        (Here grid is 2D and selection is a 2D mask.)
        """
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, n_cols = grid.shape
        total = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if grid[i, j] == color:
                    total += 1
        if total == 0:
            # Instead of using advanced indexing, use explicit loops.
            for i in range(n_rows):
                for j in range(n_cols):
                    if selection[i, j]:
                        grid_3d[0, i, j] = color
            return grid_3d
        else:
            out = np.empty((1, n_rows, n_cols), dtype=grid.dtype)
            for i in range(n_rows):
                for j in range(n_cols):
                    out[0, i, j] = grid[i, j]
            return out

    def color(self, grid, selection, method, param):
        """
        Apply a color transformation: set every selected cell to the color computed by select_color.
        """
        color_selected = select_color_numba(grid, method, param)
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, n_cols = grid.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if selection[i, j]:
                    grid_3d[0, i, j] = color_selected
        return grid_3d

    def fill_with_color(self, grid, selection, method, param):
        grid_3d = create_grid3d_numba(grid, selection)
        fill_color = select_color_numba(grid, method, param)
        if not (fill_color >= 0 and fill_color < 10):
            return grid_3d
        filled = binary_fill_holes_numba(selection)
        n_rows, n_cols = selection.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if filled[i, j] and (not selection[i, j]):
                    grid_3d[0, i, j] = fill_color
        return grid_3d

    def fill_bounding_rectangle_with_color(self, grid, selection, method, param):
        color = select_color_numba(grid, method, param)
        grid_3d = create_grid3d_numba(grid, selection)
        bound = find_bounding_rectangle_numba(selection)
        n_rows, n_cols = selection.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if bound[i, j] and (not selection[i, j]):
                    grid_3d[0, i, j] = color
        return grid_3d

    def fill_bounding_square_with_color(self, grid, selection, method, param):
        color = select_color_numba(grid, method, param)
        grid_3d = create_grid3d_numba(grid, selection)
        bound = find_bounding_square_numba(selection)
        n_rows, n_cols = selection.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if bound[i, j] and (not selection[i, j]):
                    grid_3d[0, i, j] = color
        return grid_3d

    # --- Flipping Transformations ---
    def flipv(self, grid, selection):
        """
        Flip the grid vertically (top-to-bottom) within the bounding rectangle.
        """
        grid_3d = create_grid3d_numba(grid, selection)
        bounding_rectangle = find_bounding_rectangle_numba(selection)
        n_rows, n_cols = selection.shape
        # Compute the vertically flipped grid.
        flipped = np.empty_like(grid_3d)
        for i in range(n_rows):
            for j in range(n_cols):
                flipped[0, i, j] = grid_3d[0, n_rows - 1 - i, j]
        for i in range(n_rows):
            for j in range(n_cols):
                if bounding_rectangle[i, j]:
                    grid_3d[0, i, j] = flipped[0, i, j]
        return grid_3d

    def fliph(self, grid, selection):
        """
        Flip the grid horizontally (left-to-right) within the bounding rectangle.
        """
        grid_3d = create_grid3d_numba(grid, selection)
        bounding_rectangle = find_bounding_rectangle_numba(selection)
        n_rows, n_cols = selection.shape
        flipped = np.empty_like(grid_3d)
        for i in range(n_rows):
            for j in range(n_cols):
                flipped[0, i, j] = grid_3d[0, i, n_cols - 1 - j]
        for i in range(n_rows):
            for j in range(n_cols):
                if bounding_rectangle[i, j]:
                    grid_3d[0, i, j] = flipped[0, i, j]
        return grid_3d

    def flip_main_diagonal(self, grid, selection):
        """
        Mirror the selected region along the main diagonal (transpose).
        Only operates on the bounding square.
        """
        grid_3d = create_grid3d_numba(grid, selection)
        bounding_square = find_bounding_square_numba(selection)
        n_rows, n_cols = selection.shape
        # Use the bounding square to extract the region.
        rows, cols = np.where(bounding_square)
        if rows.size > 0 and cols.size > 0 and ((rows.max() - rows.min()) == (cols.max() - cols.min())):
            min_row = rows.min()
            max_row = rows.max()
            min_col = cols.min()
            max_col = cols.max()
            # Extract the sub-grid and transpose it.
            sub_grid = np.empty((max_row - min_row + 1, max_col - min_col + 1), dtype=grid.dtype)
            for i in range(max_row - min_row + 1):
                for j in range(max_col - min_col + 1):
                    sub_grid[i, j] = grid_3d[0, min_row + i, min_col + j]
            # Transpose (mirror along main diagonal)
            for i in range(max_row - min_row + 1):
                for j in range(max_col - min_col + 1):
                    grid_3d[0, min_row + i, min_col + j] = sub_grid[j, i]
        return grid_3d

    def flip_anti_diagonal(self, grid, selection):
        """
        Mirror the selected region along the anti-diagonal.
        """
        grid_3d = create_grid3d_numba(grid, selection)
        bounding_square = find_bounding_square_numba(selection)
        n_rows, n_cols = selection.shape
        rows, cols = np.where(bounding_square)
        if rows.size > 0 and cols.size > 0 and ((rows.max()-rows.min()) == (cols.max()-cols.min())):
            min_row = rows.min()
            max_row = rows.max()
            min_col = cols.min()
            max_col = cols.max()
            sub_grid = np.empty((max_row - min_row + 1, max_col - min_col + 1), dtype=grid.dtype)
            for i in range(max_row - min_row + 1):
                for j in range(max_col - min_col + 1):
                    sub_grid[i, j] = grid_3d[0, min_row + i, min_col + j]
            # Compute anti-diagonal mirror by rotating 90 then flipping horizontally.
            temp = np.empty_like(sub_grid)
            size = sub_grid.shape[0]
            for i in range(size):
                for j in range(size):
                    temp[i, j] = sub_grid[size - 1 - j, i]
            mirrored = np.empty_like(temp)
            for i in range(size):
                for j in range(size):
                    mirrored[i, j] = temp[i, size - 1 - j]
            for i in range(size):
                for j in range(size):
                    grid_3d[0, min_row + i, min_col + j] = mirrored[i, j]
        return grid_3d

    # --- Rotation Transformations ---
    def rotate(self, grid, selection, num_rotations):
        grid_3d = create_grid3d_numba(grid, selection)
        bound = find_bounding_square_numba(selection)
        rows, cols = bound.shape
        rows_idx, cols_idx = np.where(bound)
        if rows_idx.size == 0 or cols_idx.size == 0:
            return grid_3d
        if (rows_idx.max()-rows_idx.min()) == (cols_idx.max()-cols_idx.min()):
            row_start = rows_idx.min()
            row_end = rows_idx.max()+1
            col_start = cols_idx.min()
            col_end = cols_idx.max()+1
            sub_grid = grid_3d[0, row_start:row_end, col_start:col_end]
            rotated_sub_grid = np.rot90(sub_grid, num_rotations)
            grid_3d[0, row_start:row_end, col_start:col_end] = rotated_sub_grid
        return grid_3d

    def rotate_90(self, grid, selection):
        return self.rotate(grid, selection, 1)

    def rotate_180(self, grid, selection):
        return self.rotate(grid, selection, 2)

    def rotate_270(self, grid, selection):
        return self.rotate(grid, selection, 3)

    # --- Mirror and Duplicate Transformations ---
    def mirror_down(self, grid, selection):
        """
        Mirror the selection vertically below the original grid.
        (Here grid is 2D and selection is a 2D mask; d is fixed to 1.)
        """
        n_rows, n_cols = grid.shape
        grid_3d = create_grid3d_numba(grid, selection)
        if n_rows > 15:
            return grid_3d
        new_grid = np.zeros((1, n_rows * 2, n_cols), dtype=grid.dtype)
        # Copy original grid into top half.
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, j] = grid_3d[0, i, j]
        # Copy vertically flipped version into bottom half.
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, n_rows + i, j] = grid_3d[0, n_rows - 1 - i, j]
        # Manually flip the selection mask vertically.
        flipped_sel = np.empty((n_rows, n_cols), dtype=np.bool_)
        for i in range(n_rows):
            for j in range(n_cols):
                flipped_sel[i, j] = selection[n_rows - 1 - i, j]
        # Zero out cells in the bottom half where the flipped selection is False.
        for i in range(n_rows):
            for j in range(n_cols):
                if not flipped_sel[i, j]:
                    new_grid[0, n_rows + i, j] = 0
        return new_grid.astype(np.int64)

    def mirror_up(self, grid, selection):
        n_rows, n_cols = grid.shape
        grid_3d = create_grid3d_numba(grid, selection)
        if n_rows > 15:
            return grid_3d
        new_grid = np.zeros((1, n_rows * 2, n_cols), dtype=grid.dtype)
        # For mirror_up, place the vertically flipped grid on top.
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, j] = grid_3d[0, n_rows - 1 - i, j]
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, n_rows + i, j] = grid_3d[0, i, j]
        flipped_sel = np.empty((n_rows, n_cols), dtype=np.bool_)
        for i in range(n_rows):
            for j in range(n_cols):
                flipped_sel[i, j] = selection[n_rows - 1 - i, j]
        for i in range(n_rows):
            for j in range(n_cols):
                if not flipped_sel[i, j]:
                    new_grid[0, i, j] = 0
        return new_grid.astype(np.int64)

    def mirror_right(self, grid, selection):
        n_rows, n_cols = grid.shape
        grid_3d = create_grid3d_numba(grid, selection)
        if n_cols > 15:
            return grid_3d
        new_grid = np.zeros((1, n_rows, n_cols * 2), dtype=grid.dtype)
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, j] = grid_3d[0, i, j]
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, n_cols + j] = grid_3d[0, i, n_cols - 1 - j]
        # Flip selection horizontally.
        flipped_sel = np.empty((n_rows, n_cols), dtype=np.bool_)
        for i in range(n_rows):
            for j in range(n_cols):
                flipped_sel[i, j] = selection[i, n_cols - 1 - j]
        for i in range(n_rows):
            for j in range(n_cols):
                if not flipped_sel[i, j]:
                    new_grid[0, i, n_cols + j] = 0
        return new_grid.astype(np.int64)

    def mirror_left(self, grid, selection):
        n_rows, n_cols = grid.shape
        grid_3d = create_grid3d_numba(grid, selection)
        if n_cols > 15:
            return grid_3d
        new_grid = np.zeros((1, n_rows, n_cols * 2), dtype=grid.dtype)
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, j] = grid_3d[0, i, n_cols - 1 - j]
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, n_cols + j] = grid_3d[0, i, j]
        flipped_sel = np.empty((n_rows, n_cols), dtype=np.bool_)
        for i in range(n_rows):
            for j in range(n_cols):
                flipped_sel[i, j] = selection[i, n_cols - 1 - j]
        for i in range(n_rows):
            for j in range(n_cols):
                if not flipped_sel[i, j]:
                    new_grid[0, i, j] = 0
        return new_grid.astype(np.int64)

    def duplicate_horizontally(self, grid, selection):
        """
        Duplicate the selection horizontally.
        """
        n_rows, n_cols = grid.shape
        if n_cols > 15:
            return create_grid3d_numba(grid, selection)
        grid_3d = create_grid3d_numba(grid, selection)
        new_grid = np.zeros((1, n_rows, n_cols*2), dtype=grid.dtype)
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, j] = grid_3d[0, i, j]
                if selection[i, j]:
                    new_grid[0, i, j+n_cols] = grid_3d[0, i, j]
        return new_grid

    def duplicate_vertically(self, grid, selection):
        """
        Duplicate the selection vertically.
        """
        n_rows, n_cols = grid.shape
        if n_rows > 15:
            return create_grid3d_numba(grid, selection)
        grid_3d = create_grid3d_numba(grid, selection)
        new_grid = np.zeros((1, n_rows*2, n_cols), dtype=grid.dtype)
        for i in range(n_rows):
            for j in range(n_cols):
                new_grid[0, i, j] = grid_3d[0, i, j]
                if selection[i, j]:
                    new_grid[0, i+n_rows, j] = grid_3d[0, i, j]
        return new_grid

    # --- Copy-Paste Transformations ---
    def copy_paste(self, grid, selection, shift_x, shift_y):
        grid_3d = create_grid3d_numba(grid, selection)
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            new_i = i + shift_y
            new_j = j + shift_x
            if new_i >= 0 and new_i < grid_3d.shape[1] and new_j >= 0 and new_j < grid_3d.shape[2]:
                grid_3d[0, new_i, new_j] = grid_3d[0, i, j]
        return grid_3d

    def copy_sum(self, grid, selection, shift_x, shift_y):
        grid_3d = create_grid3d_numba(grid, selection)
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            new_i = i + shift_y
            new_j = j + shift_x
            if new_i >= 0 and new_i < grid_3d.shape[1] and new_j >= 0 and new_j < grid_3d.shape[2]:
                grid_3d[0, new_i, new_j] = (grid_3d[0, new_i, new_j] + grid_3d[0, i, j]) % 10
        return grid_3d

    def cut_paste(self, grid, selection, shift_x, shift_y):
        grid_3d = create_grid3d_numba(grid, selection)
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            new_i = i + shift_y
            new_j = j + shift_x
            if new_i >= 0 and new_i < grid_3d.shape[1] and new_j >= 0 and new_j < grid_3d.shape[2]:
                grid_3d[0, new_i, new_j] = grid_3d[0, i, j]
            grid_3d[0, i, j] = 0
        return grid_3d

    def cut_sum(self, grid, selection, shift_x, shift_y):
        grid_3d = create_grid3d_numba(grid, selection)
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            new_i = i + shift_y
            new_j = j + shift_x
            if new_i >= 0 and new_i < grid_3d.shape[1] and new_j >= 0 and new_j < grid_3d.shape[2]:
                grid_3d[0, new_i, new_j] = (grid_3d[0, new_i, new_j] + grid_3d[0, i, j]) % 10
            grid_3d[0, i, j] = 0
        return grid_3d

    def copy_paste_vertically(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, n_cols = grid.shape
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            new_i = i - (n_rows - i)
            if new_i >= 0:
                grid_3d[0, new_i, j] = grid_3d[0, i, j]
            new_i = i + (n_rows - i)
            if new_i < n_rows:
                grid_3d[0, new_i, j] = grid_3d[0, i, j]
        return grid_3d

    def copy_paste_horizontally(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, n_cols = grid.shape
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            new_j = j - (n_cols - j)
            if new_j >= 0:
                grid_3d[0, i, new_j] = grid_3d[0, i, j]
            new_j = j + (n_cols - j)
            if new_j < n_cols:
                grid_3d[0, i, new_j] = grid_3d[0, i, j]
        return grid_3d

    # --- Gravitate Transformations (Block Moves) ---
    def gravitate_whole_downwards_paste(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, _ = grid.shape
        sel_rows = np.argwhere(selection)
        if sel_rows.shape[0] == 0:
            return grid_3d
        max_row = sel_rows[:, 0].max()
        shift = n_rows - max_row - 1
        return self.copy_paste(grid, selection, 0, shift)

    def gravitate_whole_upwards_paste(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        sel_rows = np.argwhere(selection)
        if sel_rows.shape[0] == 0:
            return grid_3d
        min_row = sel_rows[:, 0].min()
        shift = -min_row
        return self.copy_paste(grid, selection, 0, shift)

    def gravitate_whole_right_paste(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        _, n_cols = grid.shape
        sel_cols = np.argwhere(selection)
        if sel_cols.shape[0] == 0:
            return grid_3d
        max_col = sel_cols[:, 1].max()
        shift = n_cols - max_col - 1
        return self.copy_paste(grid, selection, shift, 0)

    def gravitate_whole_left_paste(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        sel_cols = np.argwhere(selection)
        if sel_cols.shape[0] == 0:
            return grid_3d
        min_col = sel_cols[:, 1].min()
        shift = -min_col
        return self.copy_paste(grid, selection, shift, 0)

    def gravitate_whole_downwards_cut(self, grid, selection):
        return self.cut_paste(grid, selection, 0, 1)

    def gravitate_whole_upwards_cut(self, grid, selection):
        return self.cut_paste(grid, selection, 0, -1)

    def gravitate_whole_right_cut(self, grid, selection):
        return self.cut_paste(grid, selection, 1, 0)

    def gravitate_whole_left_cut(self, grid, selection):
        return self.cut_paste(grid, selection, -1, 0)

    # --- Individual-Cell Gravity Transformations ---
    def down_gravity(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, _ = grid.shape
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            value = grid_3d[0, i, j]
            grid_3d[0, i, j] = 0
            new_i = i
            for r in range(i+1, n_rows):
                if grid_3d[0, r, j] != 0:
                    break
                new_i = r
            grid_3d[0, new_i, j] = value
        return grid_3d

    def up_gravity(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        n_rows, _ = grid.shape
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            value = grid_3d[0, i, j]
            grid_3d[0, i, j] = 0
            new_i = i
            for r in range(i-1, -1, -1):
                if grid_3d[0, r, j] != 0:
                    break
                new_i = r
            grid_3d[0, new_i, j] = value
        return grid_3d

    def right_gravity(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        n_cols = grid.shape[1]
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            value = grid_3d[0, i, j]
            grid_3d[0, i, j] = 0
            new_j = j
            for c in range(j+1, n_cols):
                if grid_3d[0, i, c] != 0:
                    break
                new_j = c
            grid_3d[0, i, new_j] = value
        return grid_3d

    def left_gravity(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        inds = np.argwhere(selection)
        for k in range(inds.shape[0]):
            i = inds[k, 0]
            j = inds[k, 1]
            value = grid_3d[0, i, j]
            grid_3d[0, i, j] = 0
            new_j = j
            for c in range(j-1, -1, -1):
                if grid_3d[0, i, c] != 0:
                    break
                new_j = c
            grid_3d[0, i, new_j] = value
        return grid_3d

    # --- Upscale Transformations ---
    def vupscale(self, grid, selection, scale_factor):
        grid_3d = create_grid3d_numba(grid, selection)
        orig_rows, orig_cols = grid.shape
        up_grid = np.repeat(grid_3d, scale_factor, axis=1)
        final_grid = np.empty((1, orig_rows, orig_cols), dtype=grid.dtype)
        start = (up_grid.shape[1] - orig_rows) // 2
        for i in range(orig_rows):
            for j in range(orig_cols):
                final_grid[0, i, j] = up_grid[0, start+i, j]
        return final_grid

    def hupscale(self, grid, selection, scale_factor):
        grid_3d = create_grid3d_numba(grid, selection)
        orig_rows, orig_cols = grid.shape
        up_grid = np.repeat(grid_3d, scale_factor, axis=2)
        final_grid = np.empty((1, orig_rows, orig_cols), dtype=grid.dtype)
        start = (up_grid.shape[2] - orig_cols) // 2
        for i in range(orig_rows):
            for j in range(orig_cols):
                final_grid[0, i, j] = up_grid[0, i, start+j]
        return final_grid

    # --- Delete / Crop Transformations ---
    def crop(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        rows, cols = selection.shape
        bound = find_bounding_rectangle_numba(selection)
        for i in range(rows):
            for j in range(cols):
                if not bound[i, j]:
                    grid_3d[0, i, j] = -1
        return grid_3d

    def delete(self, grid, selection):
        grid_3d = create_grid3d_numba(grid, selection)
        for i in range(selection.shape[0]):
            for j in range(selection.shape[1]):
                if selection[i, j]:
                    grid_3d[0, i, j] = 0
        return grid_3d

    def change_background_color(self, grid, selection, new_color):
        grid_3d = create_grid3d_numba(grid, selection)
        background_color = rankcolor_numba(grid, 0)
        n_rows, n_cols = grid.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if grid_3d[0, i, j] == background_color:
                    grid_3d[0, i, j] = new_color
        if not (new_color >= 0 and new_color < 10):
            return grid_3d
        return grid_3d

    def change_selection_to_background_color(self, grid, selection):
        from dsl.color_select import ColorSelector  # cannot be jitted; used externally
        color_selector = ColorSelector()
        background_color = color_selector.mostcolor(grid)
        grid_3d = create_grid3d_numba(grid, selection)
        for i in range(selection.shape[0]):
            for j in range(selection.shape[1]):
                if selection[i, j]:
                    grid_3d[0, i, j] = background_color
        return grid_3d

# =============================================================================
# Final Test Block and Running Time Tests
# =============================================================================

def main():
    grid = np.random.randint(0, 10, size=(30, 30)).astype(np.int64)
    selection = (grid == 3)  # 2D selection mask
    transformer = TransformerNumba()
    print("Numba new_color shape:", transformer.new_color(grid, selection, 5).shape)
    print("Numba flipv shape:", transformer.flipv(grid, selection).shape)
    print("Numba rotate_90 shape:", transformer.rotate_90(grid, selection).shape)
    print("Numba crop shape:", transformer.crop(grid, selection).shape)
    print("Numba mirror_down shape:", transformer.mirror_down(grid, selection).shape)
    print("Numba duplicate_horizontally shape:", transformer.duplicate_horizontally(grid, selection).shape)

def run_time_tests():
    grid = np.random.randint(0, 10, size=(1000, 1000)).astype(np.int64)
    selection = (grid == 3)  # 2D selection mask
    transformer = TransformerNumba()
    _ = transformer.new_color(grid, selection, 5)
    _ = transformer.color(grid, selection, 0, 2)
    _ = transformer.fill_with_color(grid, selection, 0, 4)
    _ = transformer.fill_bounding_rectangle_with_color(grid, selection, 0, 3)
    _ = transformer.flipv(grid, selection)
    _ = transformer.rotate_90(grid, selection)
    _ = transformer.crop(grid, selection)
    _ = transformer.delete(grid, selection)
    iterations = 100

    start = time.time()
    for _ in range(iterations):
         _ = transformer.new_color(grid, selection, 5)
    end = time.time()
    print("TransformerNumba.new_color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.color(grid, selection, 0, 2)
    end = time.time()
    print("TransformerNumba.color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.fill_with_color(grid, selection, 0, 4)
    end = time.time()
    print("TransformerNumba.fill_with_color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.fill_bounding_rectangle_with_color(grid, selection, 0, 3)
    end = time.time()
    print("TransformerNumba.fill_bounding_rectangle_with_color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.flipv(grid, selection)
    end = time.time()
    print("TransformerNumba.flipv avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.rotate_90(grid, selection)
    end = time.time()
    print("TransformerNumba.rotate_90 avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.crop(grid, selection)
    end = time.time()
    print("TransformerNumba.crop avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.delete(grid, selection)
    end = time.time()
    print("TransformerNumba.delete avg time: {:.6f} s".format((end - start) / iterations))

if __name__ == '__main__':
    main()
    run_time_tests()
























#%%

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
        # Ensure the selection is 3D
        if selection.ndim == 2:
            selection = selection[np.newaxis, ...]
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
        # Ensure the selection mask is 3D.
        if selection.ndim == 2:
            selection = selection[np.newaxis, ...]
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
        # Ensure the selection mask is 3D.
        if selection.ndim == 2:
            selection = selection[np.newaxis, ...]
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
        # Ensure the selection mask is 3D.
        if selection.ndim == 2:
            selection = selection[np.newaxis, ...]        
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
        # Ensure the selection mask is 3D.
        if selection.ndim == 2:
            selection = selection[np.newaxis, ...]        
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

    

# =============================================================================
# Final Test Block and Running Time Tests for Numpy Version
# =============================================================================

def main():
    # Create a small grid (30x30) with a 2D boolean selection mask.
    grid = np.random.randint(0, 10, size=(30, 30)).astype(np.int64)
    selection = (grid == 3)  # 2D mask
    transformer = Transformer()  # from your numpy version
    print("Numpy new_color shape:", transformer.new_color(grid, selection, 5).shape)
    print("Numpy flipv shape:", transformer.flipv(grid, selection).shape)
    print("Numpy rotate_90 shape:", transformer.rotate_90(grid, selection).shape)
    print("Numpy crop shape:", transformer.crop(grid, selection).shape)
    print("Numpy mirror_down shape:", transformer.mirror_down(grid, selection).shape)
    print("Numpy duplicate_horizontally shape:", transformer.duplicate_horizontally(grid, selection).shape)

def run_time_tests():
    # Create a large grid (1000x1000) with a 2D boolean selection mask.
    grid = np.random.randint(0, 10, size=(1000, 1000)).astype(np.int64)
    selection = (grid == 3)  # 2D mask
    transformer = Transformer()
    # Warm-up: call each method once.
    _ = transformer.new_color(grid, selection, 5)
    _ = transformer.color(grid, selection, 'color_rank', 2)
    _ = transformer.fill_with_color(grid, selection, 'color_rank', 4)
    _ = transformer.fill_bounding_rectangle_with_color(grid, selection, 'color_rank', 3)
    _ = transformer.flipv(grid, selection)
    _ = transformer.rotate_90(grid, selection)
    _ = transformer.crop(grid, selection)
    _ = transformer.delete(grid, selection)
    
    iterations = 100
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.new_color(grid, selection, 5)
    end = time.time()
    print("Transformer.new_color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.color(grid, selection, 'color_rank', 2)
    end = time.time()
    print("Transformer.color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.fill_with_color(grid, selection, 'color_rank', 4)
    end = time.time()
    print("Transformer.fill_with_color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.fill_bounding_rectangle_with_color(grid, selection, 'color_rank', 3)
    end = time.time()
    print("Transformer.fill_bounding_rectangle_with_color avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.flipv(grid, selection)
    end = time.time()
    print("Transformer.flipv avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.rotate_90(grid, selection)
    end = time.time()
    print("Transformer.rotate_90 avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.crop(grid, selection)
    end = time.time()
    print("Transformer.crop avg time: {:.6f} s".format((end - start) / iterations))
    
    start = time.time()
    for _ in range(iterations):
         _ = transformer.delete(grid, selection)
    end = time.time()
    print("Transformer.delete avg time: {:.6f} s".format((end - start) / iterations))


if __name__ == '__main__':
    main()
    run_time_tests()
