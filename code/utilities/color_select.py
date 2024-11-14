import numpy as np
from checks import check_color_rank

# Implemented methods:
# - mostcolor(grid): Return the most common color in the grid.
# - leastcolor(grid): Return the least common color in the grid.
# - rankcolor(grid, rank): Return the rank-th common color in the grid.

class ColorSelector:
    def __init__(self, num_colors: int = 9):
        self.num_colors = num_colors
        self.all_colors = np.arange(num_colors)
        self.invalid_color = -1 # Return this value if the color is invalid so the selection doesn't select anything.

    def mostcolor(self, grid: np.ndarray) -> int:
        """ most common color """
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        print(counts)
        return np.argmax(counts)
    
    def leastcolor(self, grid: np.ndarray) -> int:
        """ least common color """
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        counts[counts == 0] = 10000000
        return np.argmin(counts)
    
    def rankcolor(self, grid: np.ndarray, rank: int) -> int:
        """ rank-th common color """
        if check_color_rank(rank, self.num_colors) == False:
            return self.invalid_color
        values = grid.flatten()
        counts = np.bincount(values, minlength=self.num_colors)
        return np.argsort(-counts)[rank]
    
    