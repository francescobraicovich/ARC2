import json
from rearc.generators import * # Import all generator functions from the generators module.
from rearc.generators import generators
'''
def get_generators() -> dict:
    """
    Returns a mapping from task identifiers (keys) to example generator functions.
    For each function in the generators tuple, the key is the function name
    with the "generate_" prefix stripped.
    """
    prefix = 'generate_'
    return {
         # Strip the prefix from the function's name to create the key.
         func.__name__[len(prefix):]: func for func in generators
    }
'''

def demo_generator(key, n=6):
    """
    Generate a set of examples for a given ARC (Abstraction and Reasoning Corpus) task.
    This function loads the task data from a JSON file based on the provided key, concatenates its 'train' and 'test' sets, 
    and then uses a dynamically obtained generator function (named generate_<key>) from the generators module to create a list of examples.
    Parameters:
        key (str): The identifier for the ARC task. This is used to locate the corresponding JSON file in the 'arc_original/training' directory.
        n (int, optional): The number of examples to generate. Defaults to 6.
    Returns:
        list: A list containing 'n' generated examples for the specified ARC task.
    Note:
        The function assumes that the JSON file has both 'train' and 'test' keys. It also expects that a generator function
        named 'generate_<key>' exists within the generators module, which is used to generate the examples.
    """
    '''
    
    '''
    with open(f'../data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json') as fp:
        tasks = json.load(fp)
    # Access the specific task using its key.
    key_number = key[len('generate_'):]
    original_task = tasks[key_number]   # Replace with the desired key.
    original_task = original_task['train'] + original_task['test']
    generator = generators[f'generate_{key_number}']
    generated_examples = [generator(0, 1) for k in range(n)]
    '''plot_task(original_task)
    plot_task(generated_examples)'''
    return generated_examples