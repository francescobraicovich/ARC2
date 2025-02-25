import numpy as np
import json
from dsl.color_select import ColorSelector
from dsl.select import Selector
from dsl.transform import Transformer


"""
Design Principles
-----------------
This DSL was constructed following key design principles to ensure simplicity, flexibility, and reusability:

1. **Functional and Type-Based Design:** No custom classes; simple data structures like grids (vector of vectors of integers) and objects (sets of cells represented as tuples of color and location) are used.

2. **Abstract and Generic Functions:** Functions are written to abstract away details and work across types (e.g., grids and objects), with minimal DSL primitives specific to rare cases.

3. **Simple Function Signatures:** Functions have small argument lists and always return a single entity, promoting clarity and ease of use.

4. **Limited and Simple Types:** Only a small set of types is used (e.g., "grid", "object", "integer") to keep the DSL consistent and easy to learn.

5. **Reducing Redundancy:** DSL components that could be composed from others were avoided unless they were frequently used, leading to concise yet powerful primitives.

6. **Pruning and Expanding Components:** 
   - **Removing** underused, overly complex, or redundant components.
   - **Adding** components that simplify frequent tasks or enable significantly shorter solvers.
"""

# Note: An action is defined as (color_selection, selection, transformation).


class Solver:
    def __init__(self, path_to_challenges, dim=30, ColorSelector=ColorSelector, Selector=Selector, Transformer=Transformer):
        
        #Setup the dataset
        self.challenge_dictionary = json.load(open(path_to_challenges))
        self.dictionary_keys = list(self.challenge_dictionary.keys()) # list of keys in the dictionary
        self.num_challenges = len(self.challenge_dictionary) # number of challenges in the dictionary
        self.dim = dim # maximum dimension of the problem
        self.observation_shape = (dim, dim, 2) # shape of the grid

        # Initialize the function selectors/transformers
        self.color_selector = ColorSelector()
        self.selector = Selector()
        self.transformer = Transformer()

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------
    def setup_grid(challenge_key):
        """
        Setup the grid for the challenge
        """
        challenge = self.challenge_dictionary[challenge_key]
        challenge_test = challenge["test"]
        input = np.array(challenge_test["input"])
        output = np.array(challenge_test["output"])
        grid = np.array(input, output)
        return grid
    
    def check_correct(input, output):
        """
        Check if the output is correct
        """
        return np.array_equal(input, output)
    
    # -------------------------------------------------------------------------
    # Solvers
    # -------------------------------------------------------------------------
    def solve_007bbfb7(self):
        challenge_key = "007bbfb7"
        # Input size different from output
        pass

    def solve_00d62c1b(self, grid):
        challenge_key = "007bbfb7"
        self.setupgrid(challenge_key)
        a1 = [colsel.color_number(4), sel.select_connected_shapes_diag(), trans.fill_with_color()]
        # Insert color select as difference between input and output
        pass

    def solve_017c7c7b(self):
        # input size different from output
        pass

    def solve_025d127b(self):
        
