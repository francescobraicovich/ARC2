"""
Module defining the ARCActionSpace, a custom action space for ARC problems.
This module constructs an action space based on color selection, grid selection,
and transformation functions, and embeds them into a continuous space for use with
methods like Wolpertinger.

Dependencies:
- gymnasium (as gym)
- numpy
- functools.partial
- scikit-learn (NearestNeighbors, MinMaxScaler)
- Custom modules from dsl and utils packages
"""

# Standard library imports
from functools import partial

# Third-party imports
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Local application imports
from dsl.color_select import ColorSelector
from dsl.select import Selector
from dsl.transform import Transformer
from utils.action_space_embedding import create_approximate_similarity_matrix, mds_embed


class ARCActionSpace(Space):
    """
    A custom Gym space representing an action space for ARC tasks.
    
    The action space is composed of three parts:
        1. Color selection functions
        2. Grid selection functions
        3. Transformation functions
    
    Each part is generated from its respective dictionary (color_selection_dict,
    selection_dict, transformation_dict), and then the full action space is constructed
    as all combinations of these parts.
    """

    def __init__(self, args, ColorSelector=ColorSelector, Selector=Selector, Transformer=Transformer):
        """
        Initialize the ARCActionSpace.

        Args:
            args: Arguments containing configuration parameters (e.g., max_embedding, load_action_embedding, etc.).
            ColorSelector: Class to create color selection functions.
            Selector: Class to create grid selection functions.
            Transformer: Class to create transformation functions.
        """
        # Define the underlying space parameters
        dtype = np.float32
        shape = (3,)
        super().__init__(shape, dtype)

        # Print header for initialization logging
        SEPARATOR = '-' * 50
        print(SEPARATOR)
        print('Creating the action space')

        self.args = args

        # Initialize the function selectors/transformers
        self.color_selector = ColorSelector()
        self.selector = Selector()
        self.transformer = Transformer()

        # Define initial bounds and range values (later overridden)
        self._low_array = np.array([0, 0, 0])
        self._high_array = np.array([99, 99, 999])
        self._range_array = self._high_array - self._low_array
        self._dimensions = 3

        # Set normalized range values for action embeddings
        self._low = -1
        self._high = 1

        # Maximum and minimum embedding values (used for scaling)
        self.max_embedding = args.max_embedding
        self.min_embedding = args.min_embedding

        # Weight used when uniformising the key density
        self.OLD_KEYS_WIGHT = 1.5

        # Create dictionaries mapping keys to partial functions for each action component
        self.color_selection_dict = None
        self.create_color_selection_dict()

        self.selection_dict = None
        self.create_selection_dict()

        self.transformation_dict = None
        self.create_transformation_dict()

        # Create the complete action space from all components
        self.space = None
        self.create_action_space()

        # Load or create embedded actions
        if args.load_action_embedding:
            self.cleaned_space = np.load('src/embedded_space/cleaned_actions.npy')
            self.embedding = np.load('src/embedded_space/embedded_actions.npy')
        else:
            self.cleaned_space, self.embedding = self.embed_actions()

        print('Number of actions filtered:', len(self.cleaned_space))

        # Create the k-NN model for nearest neighbor search in the embedded space
        self.nearest_neighbors = None
        self.create_nearest_neighbors()
        print(SEPARATOR)

    def embed_actions(self):
        """
        Create an embedding for the actions by computing an approximate similarity matrix,
        performing MDS embedding, and scaling the results.

        Returns:
            cleaned_actions (np.ndarray): Array of action representations.
            embedded_actions (np.ndarray): Embedded actions in the continuous space.
        """
        args = self.args

        # Create an approximate similarity matrix of actions
        cleaned_actions, similarity_matrix = create_approximate_similarity_matrix(
            self, args.num_experiments_filter, args.filter_threshold, args.num_experiments_similarity
        )
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Embed the actions using Multi-Dimensional Scaling (MDS)
        embedded_actions = mds_embed(distance_matrix)

        # Scale the embedded actions to the range [-10, 10]
        scaler = MinMaxScaler(feature_range=(-10, 10))
        embedded_actions = scaler.fit_transform(embedded_actions)

        # Save the embedded actions for future use
        np.save('src/embedded_space/embedded_actions.npy', embedded_actions)
        np.save('src/embedded_space/cleaned_actions.npy', cleaned_actions)

        return cleaned_actions, embedded_actions

    def create_nearest_neighbors(self):
        """
        Create and fit the NearestNeighbors model using the embedded actions.
        """
        self.nearest_neighbors = NearestNeighbors(n_neighbors=None, algorithm='auto')
        self.nearest_neighbors.fit(self.embedding)

        print('NearestNeighbors model created with {} neighbors'.format(self.nearest_neighbors.n_neighbors))
        print

    def search_point(self, query_actions, k=5):
        """
        Find the k-nearest neighbors for the given query actions.

        Args:
            query_actions (array-like): A 2D array of query actions (shape: [n_queries, action_dim]).
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            distances (np.ndarray): Distances for the k-nearest neighbors for each query action.
            indices (np.ndarray): Indices of the k-nearest neighbors for each query action.
            actions (np.ndarray): Original action values for the k-nearest neighbors.
            embedded_actions (np.ndarray): Embedded representations of the k-nearest neighbors.
        """
        if self.nearest_neighbors is None:
            raise ValueError("NearestNeighbors model is not initialized. Call create_nearest_neighbors() first.")

        # Ensure the query actions are a 2D numpy array of type float32
        query_actions = np.array(query_actions, dtype=np.float32)
        if query_actions.ndim == 1:
            query_actions = query_actions.reshape(1, -1)

        # Query the k-nearest neighbors
        distances, indices = self.nearest_neighbors.kneighbors(query_actions, n_neighbors=k)
        actions = np.array([self.cleaned_space[indices[i]] for i in range(len(indices))])
        embedded_actions = np.array([self.embedding[indices[i]] for i in range(len(indices))])

        # If only a single query was provided, return a flattened result
        if query_actions.shape[0] == 1:
            actions = actions[0]
            embedded_actions = embedded_actions[0]
        return distances, indices, actions, embedded_actions

    def __call__(self):
        """
        Allows the instance to be called as a function, returning the action space.
        """
        return self.space

    def __len__(self):
        """
        Return the number of actions in the action space.
        """
        return len(self.space)

    def uniformise_density(self, dictionary):
        """
        Adjust the keys in the given dictionary so they are uniformly distributed.

        This uniformisation is important for methods like Wolpertinger
        to better learn the structure of the action space.

        Args:
            dictionary (dict): Dictionary with integer keys mapping to partial functions.

        Returns:
            new_dict (dict): Dictionary with uniformly distributed integer keys.
        """
        old_keys = list(dictionary.keys())
        sorted_indices = np.argsort(old_keys)
        maximum_key = old_keys[sorted_indices[-1]]
        minimum_key = old_keys[sorted_indices[0]]
        max_digits = len(str(maximum_key))

        # Determine the linspace extremes
        maximum = 10 ** max_digits - 1
        minimum = 0

        # Scale the original keys to span the entire linspace
        keys = (np.array(old_keys, dtype=int) - minimum_key) * maximum / (maximum_key - minimum_key)
        sorted_keys = np.sort(keys)

        # Create the new uniformly spaced keys and blend them with the original keys
        linspace = np.linspace(minimum, maximum, len(keys), dtype=int)
        new_keys = (linspace + sorted_keys * self.OLD_KEYS_WIGHT) / (1 + self.OLD_KEYS_WIGHT)
        new_keys = new_keys.astype(int)

        # Build the new dictionary with the adjusted keys
        new_dict = {}
        for i, _ in enumerate(keys):
            j = sorted_indices[i]
            original_key = old_keys[j]
            new_dict[int(new_keys[i])] = dictionary[original_key]
        return new_dict

    def create_color_selection_dict(self):
        """
        Create a dictionary of color selection functions.

        Returns:
            updated_dict (dict): Dictionary mapping normalized keys (np.float32) to partial functions for color selection.
        """
        color_selection_dict = {}
        ranks_to_consider = [0, 1, 2, 3, 9]

        for i in range(10):
            param = i

            # NOTE: Bypass the color selection using the color number since it doesn't make the agent generalize over different problems.
            # FIXME: Currently the model cannot add a new color to the grid, it can only change the current geometries.

            # Uncomment the line below if using direct color number selection.
            # color_selection_dict[0 + i] = partial(self.color_selector.color_number, color=param)

            if i in ranks_to_consider:
                # Use partial to fix the parameter for the rank-based color selection functions.
                color_selection_dict[10 + i] = partial(self.color_selector.rankcolor, rank=param)
                color_selection_dict[80 + i] = partial(self.color_selector.rank_largest_shape_color_nodiag, rank=param)
                # Uncomment the following line if needed:
                # color_selection_dict[90 + i] = partial(self.color_selector.rank_largest_shape_color_diag, rank=param)

        # Uniformise the density of the keys
        uniformised_dict = self.uniformise_density(color_selection_dict)
        updated_dict = {
            np.float32(key / (self._range_array[0] / 2) - 1): value
            for key, value in uniformised_dict.items()
        }
        self.color_selection_dict = updated_dict
        return updated_dict

    def create_selection_dict(self):
        """
        Create a dictionary of grid selection functions.

        Returns:
            updated_dict (dict): Dictionary mapping normalized keys (np.float32) to partial functions for grid selection.
        """
        selection_dict = {}
        selection_dict[00] = partial(self.selector.select_all_grid)  # Select the entire grid
        selection_dict[10] = partial(self.selector.select_color)  # Select elements by color
        selection_dict[20] = partial(self.selector.select_connected_shapes)  # Select connected shapes
        selection_dict[25] = partial(self.selector.select_connected_shapes_diag)  # Diagonally connected shapes
        selection_dict[35] = partial(self.selector.select_outer_border)  # Select the outer border
        selection_dict[40] = partial(self.selector.select_outer_border_diag)  # Diagonally select the outer border
        selection_dict[50] = partial(self.selector.select_inner_border)  # Select the inner border
        selection_dict[55] = partial(self.selector.select_inner_border_diag)  # Diagonally select the inner border
        selection_dict[65] = partial(self.selector.select_adjacent_to_color, points_of_contact=1)  # Adjacent elements (1 contact)
        selection_dict[67] = partial(self.selector.select_adjacent_to_color, points_of_contact=2)
        selection_dict[69] = partial(self.selector.select_adjacent_to_color, points_of_contact=3)
        selection_dict[71] = partial(self.selector.select_adjacent_to_color, points_of_contact=4)
        # Uncomment if diagonal adjacent selection is needed:
        # selection_dict[73] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=1)
        # selection_dict[80] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=2)
        # selection_dict[82] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=3)
        # selection_dict[84] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=4)
        # selection_dict[86] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=5)
        # selection_dict[88] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=6)
        # selection_dict[90] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=7)
        # selection_dict[92] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=8)
        selection_dict[99] = partial(self.selector.select_all_grid)

        # Uniformise the density of the keys
        uniformised_dict = self.uniformise_density(selection_dict)
        updated_dict = {
            np.float32(key / (self._range_array[1] / 2) - 1): value
            for key, value in uniformised_dict.items()
        }
        self.selection_dict = updated_dict
        return updated_dict

    def create_transformation_dict(self):
        """
        Create a dictionary of transformation functions.

        Returns:
            updated_dict (dict): Dictionary mapping normalized keys (np.float32) to partial functions for transformations.
        """
        transformation_dict = {}

        # Coloring transformations: assign new colors using a step size of 5 starting at -50.
        new_color_start = -50
        for i in range(10):
            transformation_dict[new_color_start + i * 5] = partial(self.transformer.new_color, color=i)

        # Various color and shape based transformations with different starting keys and displacements.
        color_start = 0
        fill_with_color_start = 50
        fill_bounding_rectangle_with_color_start = 100
        fill_bounding_square_with_color_start = 150
        for i in [0, 1, 2, 9]:
            if i == 0:
                displacement = 0
            elif i == 1:
                displacement = 2
            elif i == 2:
                displacement = 4
            elif i == 9:
                displacement = 6

            # Transformations by color rank
            transformation_dict[color_start + displacement] = partial(
                self.transformer.color, method='color_rank', param=i
            )
            transformation_dict[fill_with_color_start + displacement] = partial(
                self.transformer.fill_with_color, method='color_rank', param=i
            )
            transformation_dict[fill_bounding_rectangle_with_color_start + displacement] = partial(
                self.transformer.fill_bounding_rectangle_with_color, method='color_rank', param=i
            )
            transformation_dict[fill_bounding_square_with_color_start + displacement] = partial(
                self.transformer.fill_bounding_square_with_color, method='color_rank', param=i
            )

            # Transformations by shape rank (nodiag)
            transformation_dict[color_start + displacement + 10] = partial(
                self.transformer.color, method='shape_rank_nodiag', param=i
            )
            transformation_dict[fill_with_color_start + displacement + 10] = partial(
                self.transformer.fill_with_color, method='shape_rank_nodiag', param=i
            )
            transformation_dict[fill_bounding_rectangle_with_color_start + displacement + 10] = partial(
                self.transformer.fill_bounding_rectangle_with_color, method='shape_rank_nodiag', param=i
            )
            transformation_dict[fill_bounding_square_with_color_start + displacement + 10] = partial(
                self.transformer.fill_bounding_square_with_color, method='shape_rank_nodiag', param=i
            )

            # Uncomment the following block to include diagonal shape rank transformations:
            # transformation_dict[color_start + displacement + 20] = partial(
            #     self.transformer.color, method='shape_rank_diag', param=i
            # )
            # transformation_dict[fill_with_color_start + displacement + 20] = partial(
            #     self.transformer.fill_with_color, method='shape_rank_diag', param=i
            # )
            # transformation_dict[fill_bounding_rectangle_with_color_start + displacement + 20] = partial(
            #     self.transformer.fill_bounding_rectangle_with_color, method='shape_rank_diag', param=i
            # )
            # transformation_dict[fill_bounding_square_with_color_start + displacement + 20] = partial(
            #     self.transformer.fill_bounding_square_with_color, method='shape_rank_diag', param=i
            # )

        # Flipping transformations
        transformation_dict[200] = partial(self.transformer.flipv)
        transformation_dict[210] = partial(self.transformer.fliph)
        transformation_dict[220] = partial(self.transformer.flip_main_diagonal)
        transformation_dict[230] = partial(self.transformer.flip_anti_diagonal)

        # Rotating transformations
        transformation_dict[270] = partial(self.transformer.rotate_90)
        transformation_dict[280] = partial(self.transformer.rotate_180)
        transformation_dict[290] = partial(self.transformer.rotate_270)

        # Mirroring and duplicating transformations
        transformation_dict[314] = partial(self.transformer.mirror_left)
        transformation_dict[322] = partial(self.transformer.mirror_right)
        transformation_dict[330] = partial(self.transformer.mirror_up)
        transformation_dict[338] = partial(self.transformer.mirror_down)
        transformation_dict[360] = partial(self.transformer.duplicate_horizontally)
        transformation_dict[370] = partial(self.transformer.duplicate_vertically)

        # Copy/Cut and paste transformations
        transformation_dict[400] = partial(self.transformer.copy_paste_vertically)
        transformation_dict[410] = partial(self.transformer.copy_paste_horizontally)

        # Parameters for copy/cut and paste/sum operations with shifts
        copy_paste_start = 420
        cut_paste_start = 480
        copy_sum_start = 540
        cut_sum_start = 600

        # Use nested loops with steps of 2 for shift parameters (to limit the number of transformations)
        for i in range(1, 5, 2):
            for j in range(1, 5, 2):
                key_offset = i * j * 3
                transformation_dict[copy_paste_start + key_offset] = partial(
                    self.transformer.copy_paste, shift_x=i, shift_y=j
                )
                transformation_dict[cut_paste_start + key_offset] = partial(
                    self.transformer.cut_paste, shift_x=i, shift_y=j
                )
                transformation_dict[copy_sum_start + key_offset] = partial(
                    self.transformer.copy_sum, shift_x=i, shift_y=j
                )
                transformation_dict[cut_sum_start + key_offset] = partial(
                    self.transformer.cut_sum, shift_x=i, shift_y=j
                )

        # Gravitate transformations
        transformation_dict[700] = partial(self.transformer.gravitate_whole_upwards_paste)
        transformation_dict[710] = partial(self.transformer.gravitate_whole_downwards_paste)
        transformation_dict[720] = partial(self.transformer.gravitate_whole_right_paste)
        transformation_dict[730] = partial(self.transformer.gravitate_whole_left_paste)
        transformation_dict[750] = partial(self.transformer.gravitate_whole_upwards_cut)
        transformation_dict[760] = partial(self.transformer.gravitate_whole_downwards_cut)
        transformation_dict[770] = partial(self.transformer.gravitate_whole_right_cut)
        transformation_dict[780] = partial(self.transformer.gravitate_whole_left_cut)
        transformation_dict[800] = partial(self.transformer.up_gravity)
        transformation_dict[810] = partial(self.transformer.down_gravity)
        transformation_dict[820] = partial(self.transformer.right_gravity)
        transformation_dict[830] = partial(self.transformer.left_gravity)

        # Upscaling transformations
        transformation_dict[890] = partial(self.transformer.vupscale, scale_factor=2)
        transformation_dict[900] = partial(self.transformer.hupscale, scale_factor=3)
        transformation_dict[920] = partial(self.transformer.vupscale, scale_factor=2)
        transformation_dict[930] = partial(self.transformer.hupscale, scale_factor=3)

        # Deletion and cropping transformations
        transformation_dict[970] = partial(self.transformer.crop)
        transformation_dict[999] = partial(self.transformer.delete)

        # Uniformise the density of the transformation keys
        uniformised_dict = self.uniformise_density(transformation_dict)
        updated_dict = {
            np.float32(key / (self._range_array[2] / 2) - 1): value
            for key, value in uniformised_dict.items()
        }
        self.transformation_dict = updated_dict
        return updated_dict

    def create_action_space(self):
        """
        Build the complete action space as an array of action vectors, where each action
        is represented by a 3-dimensional vector: [color_key, selection_key, transformation_key].

        Returns:
            action_space (list): List of action vectors.
        """
        action_space = []

        # Generate all combinations of color, selection, and transformation keys
        for color_key in self.color_selection_dict.keys():
            for selection_key in self.selection_dict.keys():
                for transformation_key in self.transformation_dict.keys():
                    action = np.zeros(3, dtype=np.float32)
                    action[0] = color_key
                    action[1] = selection_key
                    action[2] = transformation_key
                    action_space.append(action)

        print(f"Number of actions not filtered: {len(action_space)}")
        self.space = np.array(action_space)
        return action_space

    def get_space(self):
        """
        Return the full action space.

        Returns:
            np.ndarray: The array of all possible actions.
        """
        return self.space

    def shape(self):
        """
        Return the shape of the action space.

        Returns:
            tuple: Shape of the action space array.
        """
        return self.space.shape

    def get_number_of_actions(self):
        """
        Return the total number of actions available.

        Returns:
            int: Number of actions.
        """
        return self.shape()[0]

    def action_to_string(self, action, only_color=False, only_selection=False, only_transformation=False):
        """
        Convert an action vector into a human-readable string describing each component.

        Args:
            action (array-like): The action vector (of length 3).
            only_color (bool): If True, return only the color selection string.
            only_selection (bool): If True, return only the grid selection string.
            only_transformation (bool): If True, return only the transformation string.

        Returns:
            dict or str: A dictionary with keys 'color_selection', 'selection', and 'transformation'
                         if no specific flag is provided; otherwise, the corresponding string.
        """
        # Extract each component from the action vector
        color_selection = np.float32(action[0])
        selection = np.float32(action[1])
        transformation = np.float32(action[2])

        # Retrieve and format the color selection function
        color_selection_partial = self.color_selection_dict[color_selection]
        func_name = color_selection_partial.func.__name__
        args_str = ', '.join(f'{k}={v}' for k, v in color_selection_partial.keywords.items())
        color_selection_string = f"{func_name}({args_str})"

        # Retrieve and format the grid selection function
        selection_partial = self.selection_dict[selection]
        func_name = selection_partial.func.__name__
        args_str = ', '.join(f'{k}={v}' for k, v in selection_partial.keywords.items())
        selection_string = f"{func_name}({args_str})"

        # Retrieve and format the transformation function
        transformation_partial = self.transformation_dict[transformation]
        func_name = transformation_partial.func.__name__
        args_str = ', '.join(f'{k}={v}' for k, v in transformation_partial.keywords.items())
        transformation_string = f"{func_name}({args_str})"

        action_dict = {
            'color_selection': color_selection_string,
            'selection': selection_string,
            'transformation': transformation_string
        }

        if only_color:
            return action_dict['color_selection']
        if only_selection:
            return action_dict['selection']
        if only_transformation:
            return action_dict['transformation']

        return action_dict
