import os
import numpy as np
import logging
from train_test import train, test
import warnings
import json
from arg_parser import init_parser
from setproctitle import setproctitle as ptitle
from enviroment import ARC_Env
from action_space import ARCActionSpace
from utils.util import set_device
import torch
from utils.util import get_output_folder, setup_logger
from wolp_agent import WolpertingerAgent


if __name__ == "__main__":

    device = set_device()
    print('Using device: {}'.format(device))


    ptitle('WOLP_DDPG')
    warnings.filterwarnings('ignore')
    parser = init_parser('WOLP_DDPG')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]

    
    args.save_model_dir = get_output_folder('../output', args.env)
    challenge_dictionary = json.load(open('data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json'))

    action_space_args = {
        'num_experiments_filter': args.num_experiments_filter, 
        'filter_threshold': args.filter_threshold, 
        'num_experiments_similarity': args.num_experiments_similarity,
        'load': args.load_action_embedding
    }

    action_space = ARCActionSpace(args)
    env = ARC_Env(challenge_dictionary, action_space)
    continuous = None

    # discrete action for 1 dimension
    # TODO: change the nb_states to the shape of the grid
    nb_states = 1805  # 60 x 30 grid ravels to 1800 + 4 for the dimensions of the grid + 1 for cls embedding
    nb_actions = 20  # the dimension of actions, usually it is 1. Depend on the environment.
    continuous = False

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)
  
    agent_args = {
        'nb_states': nb_states,
        'nb_actions': nb_actions,
        'args': args,
        'k': args.k_neighbors,
        'action_space': action_space
    }

    agent = WolpertingerAgent(**agent_args)
    
    if args.load:
        agent.load_weights(args.load_model_dir)

    if args.gpu_ids[0] >= 0 and args.gpu_nums > 0 and torch.cuda.is_available():
        agent.cuda_convert()

    # set logger, log args here
    log = {}
    if args.mode == 'train':
        setup_logger('RS_log', r'{}/RS_train_log'.format(args.save_model_dir))
    elif args.mode == 'test':
        setup_logger('RS_log', r'{}/RS_test_log'.format(args.save_model_dir))
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    
    log['RS_log'] = logging.getLogger('RS_log')
    d_args = vars(args)
    d_args['max_actions'] = args.max_actions
    
    for key in agent_args.keys():
        if key == 'args':
            continue
        d_args[key] = agent_args[key]
    for k in d_args.keys():
        log['RS_log'].info('{0}: {1}'.format(k, d_args[k]))

    if args.mode == 'train':
        print('Training')

        train_args = {
            'continuous': continuous,
            'env': env,
            'agent': agent,
            'max_episode': args.max_episode,
            'max_actions': args.max_actions,
            'warmup': args.warmup,
            'save_model_dir': args.save_model_dir,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            'save_per_epochs': args.save_per_epochs
        }

        train(**train_args)

    elif args.mode == 'test':

        test_args = {
            'env': env,
            'agent': agent,
            'model_path': args.load_model_dir,
            'test_episode': args.test_episode,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
        }

        test(**test_args)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))