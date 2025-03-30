#%%
import numpy as np
import numba as nb
from numba.typed import List, Dict
from numba.core.types import int64, boolean
import time

# ------------------------------------------------------------
# Utility functions for geometry selection (Numba version)
# ------------------------------------------------------------

@nb.njit
def is_valid_geometry(color_mask, start_row, start_col, end_row, end_col):
    """
    Check if the sub-array specified by the rectangle
    [start_row:end_row, start_col:end_col] is completely True.
    Also, ensure the indices are within bounds.
    """
    rows, cols = color_mask.shape
    if end_row > rows or end_col > cols:
        return False
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            if not color_mask[i, j]:
                return False
    return True

@nb.njit
def find_matching_geometries(color_mask, height, width):
    """
    Find all geometries (as 4‐tuples: (start_row, start_col, end_row, end_col))
    matching the specified dimensions that are completely True in color_mask.
    Returns a typed List of 4‐tuples.
    """
    nrows, ncols = color_mask.shape
    matching = List()  # will hold tuples (int64, int64, int64, int64)
    for i in range(nrows - height + 1):
        for j in range(ncols - width + 1):
            if is_valid_geometry(color_mask, i, j, i + height, j + width):
                matching.append((i, j, i + height, j + width))
    return matching

@nb.njit
def geometries_overlap(geo1, geo2):
    """
    Check if two geometries (each given as (r1, c1, r1_end, c1_end))
    overlap.
    """
    r1, c1, r1_end, c1_end = geo1
    r2, c2, r2_end, c2_end = geo2
    row_overlap = (r1 <= r2 and r2 < r1_end) or (r2 <= r1 and r1 < r2_end)
    col_overlap = (c1 <= c2 and c2 < c1_end) or (c2 <= c1 and c1 < c2_end)
    return row_overlap and col_overlap

# --- Recursive function to find non‐overlapping combinations.
# Because recursion is not supported in full nopython mode, we use forceobj=True.
@nb.njit(forceobj=True)
def recursive_combinations_finder(free_array, current_combinations):
    """
    free_array: 2D boolean array of shape (N, N) where free_array[i,j] is True
                if geometry i and j do NOT overlap.
    current_combinations: a typed List of Dicts (acting as sets of int64 indices).
    Recursively adds new indices that are free (non-overlapping) with all indices in each combination.
    """
    new_combinations = List()
    for comb in current_combinations:
        # Count number of indices already in this combination.
        n_comb = 0
        for _ in comb.keys():
            n_comb += 1
        # Create an array to hold the indices.
        indices = np.empty(n_comb, dtype=nb.int64)
        k = 0
        for x in comb.keys():
            indices[k] = x
            k += 1
        # Compute the "free row": for each candidate index j, check if it is free with all in comb.
        free_row = np.ones(free_array.shape[1], dtype=nb.int64)
        for i in range(indices.shape[0]):
            for j in range(free_array.shape[1]):
                if not free_array[indices[i], j]:
                    free_row[j] = 0
        # Exclude indices already in the combination.
        for i in range(indices.shape[0]):
            free_row[indices[i]] = 0

        # For every index j that is free, add it to the combination.
        for j in range(free_row.shape[0]):
            if free_row[j] != 0:
                new_set = Dict.empty(key_type=int64, value_type=boolean)
                for x in comb.keys():
                    new_set[x] = True
                new_set[j] = True
                new_combinations.append(new_set)
    if len(new_combinations) > 0:
        sub_combinations = recursive_combinations_finder(free_array, new_combinations)
        for s in new_combinations:
            current_combinations.append(s)
        for s in sub_combinations:
            current_combinations.append(s)
    return current_combinations


@nb.njit(parallel=True)
def find_non_overlapping_combinations_iter(geometries):
    n = len(geometries)
    # Precompute a free_array: free_array[i,j] is True if geometry i and j do NOT overlap.
    free_array = np.empty((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i < j:
                free_array[i, j] = not geometries_overlap(geometries[i], geometries[j])
            else:
                free_array[i, j] = False

    # We'll build combinations iteratively.
    # Start with each geometry in its own combination.
    combos = List()
    for i in range(n):
        # We store each combination as a 1D array of indices.
        a = np.empty(1, dtype=np.int64)
        a[0] = i
        combos.append(a)

    changed = True
    while changed:
        changed = False
        new_combos = List()
        # Try to add a new index to each current combination.
        for comb in combos:
            # For each candidate index not already in comb, check if it is free with every element in comb.
            for candidate in range(n):
                found = False
                for idx in range(comb.shape[0]):
                    # Check free_array for pair (comb[idx], candidate). Since free_array is defined for i<j,
                    # we must handle order.
                    a = comb[idx]
                    if a < candidate:
                        if not free_array[a, candidate]:
                            found = True
                            break
                    elif a > candidate:
                        if not free_array[candidate, a]:
                            found = True
                            break
                if not found:
                    # Candidate is free with all in comb; build a new combination by appending candidate.
                    # First, check if candidate is already in comb.
                    already_in = False
                    for idx in range(comb.shape[0]):
                        if comb[idx] == candidate:
                            already_in = True
                            break
                    if not already_in:
                        # Create new combination array with one extra element.
                        new_len = comb.shape[0] + 1
                        new_arr = np.empty(new_len, dtype=np.int64)
                        for k in range(comb.shape[0]):
                            new_arr[k] = comb[k]
                        new_arr[new_len - 1] = candidate
                        new_combos.append(new_arr)
                        changed = True
        # Append all new combinations to combos.
        for arr in new_combos:
            combos.append(arr)
    return combos




# ------------------------------------------------------------
# Testing functions for the Numba code
# ------------------------------------------------------------

def test_geometry_functions():
    # Create a random boolean grid (simulate a color mask)
    np.random.seed(0)
    mask = np.random.randint(0, 2, size=(50, 50)).astype(np.bool_)
    
    valid = is_valid_geometry(mask, 10, 10, 20, 20)
    print("is_valid_geometry (10,10,20,20):", valid)
    
    matches = find_matching_geometries(mask, 5, 5)
    print("Number of matching geometries (5x5):", len(matches))
    
    geo1 = (10, 10, 20, 20)
    geo2 = (15, 15, 25, 25)
    overlap = geometries_overlap(geo1, geo2)
    print("geometries_overlap (geo1 vs geo2):", overlap)
    
    # Test find_non_overlapping_combinations with a small list of geometries.
    geos = [(10, 10, 20, 20), (15, 15, 25, 25), (30, 30, 40, 40)]
    non_overlap = find_non_overlapping_combinations(geos)
    # Convert the typed List to a Python list of lists for printing.
    non_overlap_list = [list(arr) for arr in non_overlap]
    print("Non-overlapping combinations:", non_overlap_list)

def run_time_tests():
    # Create a large random boolean mask (simulate a color mask)
    mask = np.random.randint(0, 2, size=(500, 500)).astype(np.bool_)
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        _ = find_matching_geometries(mask, 5, 5)
    end = time.time()
    print("find_matching_geometries average time: {:.6f} s".format((end - start) / iterations))
    
    geos = find_matching_geometries(mask, 5, 5)
    # Use at most 10 geometries to keep combinations manageable.
    if len(geos) > 10:
        geos = geos[:10]
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        _ = find_non_overlapping_combinations(geos)
    end = time.time()
    print("find_non_overlapping_combinations average time: {:.6f} s".format((end - start) / iterations))

if __name__ == '__main__':
    test_geometry_functions()
    run_time_tests()














#%%
import numpy as np

# This file contains utility methods for selecting geometries in a color mask.

# Implemented utility methods:
# - is_valid_geometry(color_mask, start_row, start_col, end_row, end_col): Check if the specified geometry is valid and fully colored.
# - find_matching_geometries(color_mask, height, width): Find all geometries matching the specified dimensions in the color mask.
# - geometries_overlap(geo1, geo2): Check if two geometries overlap.
# - find_non_overlapping_combinations(geometries): Find all possible combinations of non-overlapping geometries.

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