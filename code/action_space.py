import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
from functools import partial


class ARCActionSpace(Space):
    def __init__(self, ColorSelector, Selector, Transformer):
        dtype = np.int32
        shape = (3,)
        super().__init__(shape, dtype)
        
        # Define the classes that will be used to create the action space
        self.color_selector = ColorSelector
        self.selector = Selector
        self.transformer = Transformer

        # Define the weights for the old keys when uniformising the density of the keys
        # Uniformising the density is important for Wolpertinger to learn the action space better
        self.old_keys_weight = 1.5

        self.color_selection_dict = None
        self.create_color_selection_dict()

        self.selection_dict = None
        self.create_selection_dict()

        self.transformation_dict = None
        self.create_transformation_dict()

        self.space = None
        self.create_action_space()

    def __call__(self):
        return self.space
    
    def __len__(self):
        return len(self.space)

    def uniformise_density(self, dictionary):
        """
        Makes the keys of the dictionary to be uniformly distributed
        """
        old_keys = list(dictionary.keys())
        sorted_indices = np.argsort(old_keys)
        maximum_key = old_keys[sorted_indices[-1]]
        minimum_key = old_keys[sorted_indices[0]]
        max_digits = len(str(maximum_key))

        # find the extremes of the linspace
        maximum = 10**max_digits - 1
        minimum = 0

        # scale the keys to use the whole linspace
        keys = (np.array(old_keys, dtype=int) - minimum_key) * maximum / (maximum_key - minimum_key)
        sorted_keys = np.sort(keys)

        # create the linspace
        linspace = np.linspace(minimum, maximum, len(keys), dtype=int)
        new_keys = (linspace + sorted_keys * self.old_keys_weight)/ (1 + self.old_keys_weight)
        new_keys = new_keys.astype(int)

        # create the new dictionary
        new_dict = {}
        for i, key in enumerate(keys):
            j = sorted_indices[i]
            key = old_keys[j]
            new_dict[int(new_keys[i])] = dictionary[key]
        return new_dict

    def create_color_selection_dict(self):
        color_selection_dict = {}
        ranks_to_consider = [0, 1, 2, 9]

        for i in range(10):
            param = i
            
            #NOTE: Bypass the color selection using the color number since it doesn't make the agent generalize over different problems
            #FIXME: Currently the model cannot add a new color to the grid, it can only change the current geometries.


            #color_selection_dict[0+i] = partial(self.color_selector.color_number, color=param) # Select the color with the number 'param'
            
            if i in ranks_to_consider:
                # Use partial to fix the 'param' or 'color' parameter
                color_selection_dict[50+i] = partial(self.color_selector.rankcolor, rank=param) # Select the color with the rank i
                color_selection_dict[80+i] = partial(self.color_selector.rank_largest_shape_color_nodiag, rank=param) # Select the color of the rank-th largest shape
                color_selection_dict[90+i] = partial(self.color_selector.rank_largest_shape_color_diag, rank=param) # Select the color of the rank-th largest shape considering diagonal connections

        self.color_selection_dict = self.uniformise_density(color_selection_dict)
        return color_selection_dict
    
    def create_selection_dict(self):
        selection_dict = {}
        selection_dict[00] = partial(self.selector.select_all_grid) # Select the entire grid
        selection_dict[10] = partial(self.selector.select_color) # Select the color
        selection_dict[20] = partial(self.selector.select_connected_shapes) # Select the connected shapes
        selection_dict[25] = partial(self.selector.select_connected_shapes_diag) # Select the connected shapes with diagonal connectivity
        selection_dict[35] = partial(self.selector.select_outer_border) # Select the outer border
        selection_dict[40] = partial(self.selector.select_outer_border_diag) # Select the outer border with diagonal connectivity
        selection_dict[50] = partial(self.selector.select_inner_border) # Select the inner border
        selection_dict[55] = partial(self.selector.select_inner_border_diag) # Select the inner border with diagonal connectivity
        selection_dict[65] = partial(self.selector.select_adjacent_to_color, points_of_contact=1) # Select the elements adjacent to the color
        selection_dict[67] = partial(self.selector.select_adjacent_to_color, points_of_contact=2) 
        selection_dict[69] = partial(self.selector.select_adjacent_to_color, points_of_contact=3) 
        selection_dict[71] = partial(self.selector.select_adjacent_to_color, points_of_contact=4) 
        selection_dict[73] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=1) # Select the elements adjacent to the color with diagonal connectivity
        selection_dict[80] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=2)
        selection_dict[82] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=3)
        selection_dict[84] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=4)
        selection_dict[86] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=5)
        selection_dict[88] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=6)
        selection_dict[90] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=7)
        selection_dict[92] = partial(self.selector.select_adjacent_to_color_diag, points_of_contact=8)
        selection_dict[99] = partial(self.selector.select_all_grid)

        #NOTE: This currently leaves out the selection of rectangles (only selection function with parameters)
        self.selection_dict = self.uniformise_density(selection_dict)
        return selection_dict
    
    def create_transformation_dict(self):
        transformation_dict = {}

        # Coloring transformations
        color_start = 0
        fill_with_color_start = 50
        fill_bounding_rectangle_with_color_start = 100
        fill_bounding_square_with_color_start = 150
        for i in [1, 2, 9]:
            if i == 1:
                displacement = 0
            if i == 2:
                displacement = 3
            if i == 9:
                displacement = 6

            # By color rank
            transformation_dict[color_start + displacement] = partial(self.transformer.color, method='color_rank', param = i)
            transformation_dict[fill_with_color_start + displacement] = partial(self.transformer.fill_with_color, method='color_rank', param = i)
            transformation_dict[fill_bounding_rectangle_with_color_start + displacement] = partial(self.transformer.fill_bounding_rectangle_with_color, method='color_rank', param = i)
            transformation_dict[fill_bounding_square_with_color_start + displacement] = partial(self.transformer.fill_bounding_square_with_color, method='color_rank', param = i)

            # By shape rank (nodiag)
            transformation_dict[color_start + displacement + 10] = partial(self.transformer.color, method='shape_rank_nodiag', param = i)
            transformation_dict[fill_with_color_start + displacement + 10] = partial(self.transformer.fill_with_color, method='shape_rank_nodiag', param = i)
            transformation_dict[fill_bounding_rectangle_with_color_start + displacement + 10] = partial(self.transformer.fill_bounding_rectangle_with_color, method='shape_rank_nodiag', param = i)
            transformation_dict[fill_bounding_square_with_color_start + displacement + 10] = partial(self.transformer.fill_bounding_square_with_color, method='shape_rank_nodiag', param = i)

            # By shape rank (diag)
            transformation_dict[color_start + displacement + 20] = partial(self.transformer.color, method='shape_rank_diag', param = i)
            transformation_dict[fill_with_color_start + displacement + 20] = partial(self.transformer.fill_with_color, method='shape_rank_diag', param = i)
            transformation_dict[fill_bounding_rectangle_with_color_start + displacement + 20] = partial(self.transformer.fill_bounding_rectangle_with_color, method='shape_rank_diag', param = i)
            transformation_dict[fill_bounding_square_with_color_start + displacement + 20] = partial(self.transformer.fill_bounding_square_with_color, method='shape_rank_diag', param = i)

        # Flipping transformations
        transformation_dict[200] = partial(self.transformer.flipv) # Fill with black
        transformation_dict[210] = partial(self.transformer.fliph)
        transformation_dict[220] = partial(self.transformer.flip_main_diagonal)
        transformation_dict[230] = partial(self.transformer.flip_anti_diagonal)

        # Rotating transformations
        transformation_dict[270] = partial(self.transformer.rotate_90)
        transformation_dict[280] = partial(self.transformer.rotate_180)
        transformation_dict[290] = partial(self.transformer.rotate_270)

        # Mirroring and duplicating transformations
        transformation_dict[314] = partial(self.transformer.mirror_left)
        transformation_dict[322] = partial(self.transformer.mirror_right)
        transformation_dict[330] = partial(self.transformer.mirror_up)     
        transformation_dict[338] = partial(self.transformer.mirror_down)                       
        transformation_dict[360] = partial(self.transformer.duplicate_horizontally)
        transformation_dict[370] = partial(self.transformer.duplicate_vertically)

        # Copy/Cut and paste transformations
        transformation_dict[400] = partial(self.transformer.copy_paste_vertically)
        transformation_dict[410] = partial(self.transformer.copy_paste_horizontally)
        
        copy_paste_start = 420
        cut_paste_start = 480
        copy_sum_start = 540
        cut_sum_start = 600
        for i in range(1, 5):
            for j in range(1, 5):
                transformation_dict[copy_paste_start + (i * j  * 3)] = partial(self.transformer.copy_paste, shift_x=i, shift_y=j)
                transformation_dict[cut_paste_start + (i * j  * 3)] = partial(self.transformer.cut_paste, shift_x=i, shift_y=j)
                transformation_dict[copy_sum_start + (i * j  * 3)] = partial(self.transformer.copy_sum, shift_x=i, shift_y=j)
                transformation_dict[cut_sum_start + (i * j  * 3)] = partial(self.transformer.cut_sum, shift_x=i, shift_y=j)

        # Gravitate transformations
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

        # Upscales
        transformation_dict[890] = partial(self.transformer.vupscale, scale_factor=2)
        transformation_dict[900] = partial(self.transformer.hupscale, scale_factor=3)
        transformation_dict[920] = partial(self.transformer.vupscale, scale_factor=2)
        transformation_dict[930] = partial(self.transformer.hupscale, scale_factor=3)

        # Deletion and cropping
        transformation_dict[970] = partial(self.transformer.crop)
        transformation_dict[999] = partial(self.transformer.delete)

        self.transformation_dict = self.uniformise_density(transformation_dict)
        return transformation_dict
    
    def create_action_space(self):
        action_space = []

        for i, color_key in enumerate(self.color_selection_dict.keys()):
            for j, selection_key in enumerate(self.selection_dict.keys()):
                for k, transformation_key in enumerate(self.transformation_dict.keys()):
                    action = np.zeros(3, dtype=int)
                    action[0] = color_key
                    action[1] = selection_key
                    action[2] = transformation_key
                    action_space.append(action)

        self.space = action_space
        return action_space
    
    def action_to_string(self, action, only_color=False, only_selection=False, only_transformation=False):
        color_selection = int(action[0])
        selection = int(action [1])
        transformation = int(action[2])

        # Convert the partial function to a string
        color_selection_partial = self.color_selection_dict[color_selection]
        func_name = color_selection_partial.func.__name__
        args_str = ', '.join(f'{k}={v}' for k, v in color_selection_partial.keywords.items())
        color_selection_string = f"{func_name}, {args_str}"

        # Convert the partial function to a string
        selection_partial = self.selection_dict[selection]
        func_name = selection_partial.func.__name__
        args_str = ', '.join(f'{k}={v}' for k, v in selection_partial.keywords.items())
        selection_string = f"{func_name}, {args_str}"

        # Convert the partial function to a string
        transformation_partial = self.transformation_dict[transformation]
        func_name = transformation_partial.func.__name__
        args_str = ', '.join(f'{k}={v}' for k, v in transformation_partial.keywords.items())
        transformation_string = f"{func_name}, {args_str}"

        action_dict = {'color_selection': color_selection_string, 'selection': selection_string, 'transformation': transformation_string}

        if only_color:
            return action_dict['color_selection']
        if only_selection:
            return action_dict['selection']
        if only_transformation:
            return action_dict['transformation']
        return action_dict