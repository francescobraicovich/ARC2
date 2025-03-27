import os
import warnings
import logging
import wandb
import torch
import numpy as np

from setproctitle import setproctitle as ptitle
from arg_parser_le import init_parser_le

from utils.util import (
    set_device,
    get_output_folder,
    setup_logger
)

from enviroment import ARC_Env
from action_space import ARCActionSpace
from train_le import pretrain_embedding

def main():
    warnings.filterwarnings('ignore')

    # 1. Parse arguments
    parser = init_parser_le()
    args = parser.parse_args()

    # 2. Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
    device = set_device()
    print(f'Using device: {device}')

    # 3. Optionally set a process title
    ptitle('WOLP_DDPG')

    # 4. Prepare output folder
    args.save_model_dir = get_output_folder('../output', args.env)

    # 5. Initialize wandb (only if training)
    if args.mode == 'train' and wandb.run is None:
        wandb.init(project="arc-v1", config=vars(args), mode="online")

    action_space = ARCActionSpace(args)

    # 6. Create training and evaluation environments
    train_env = ARC_Env(
        path_to_challenges='data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json',
        action_space=action_space
    )
    eval_env = ARC_Env(
        path_to_challenges='data/RAW_DATA_DIR/arc-prize-2024/arc-agi_evaluation_challenges.json',
        action_space=action_space
    )

    # 7. Set seeds
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_env.seed(args.seed)
        eval_env.seed(args.seed)

    # 8. Define state and action dimensions
    nb_states = 1805
    nb_actions = 20
    continuous = False

    # 9. Create the agent
    agent_args = {
        'nb_states': nb_states,
        'nb_actions': nb_actions,
        'args': args,
        'k': args.k_neighbors,
        'action_space': action_space
    }
    agent = WolpertingerAgent(**agent_args)

    # 10. Optionally load model weights
    if args.load:
        agent.load_weights(args.load_model_dir)

    # 12. Set up logger
    if args.mode == 'train':
        setup_logger('RS_log', f'{args.save_model_dir}/RS_train_log')
    elif args.mode == 'test':
        setup_logger('RS_log', f'{args.save_model_dir}/RS_test_log')
    else:
        raise RuntimeError(f'Undefined mode {args.mode}')
    logger = logging.getLogger('RS_log')

    # 13. Log hyperparameters
    d_args = vars(args)
    d_args['nb_states'] = nb_states
    d_args['nb_actions'] = nb_actions
    d_args['continuous'] = continuous
    for k, v in d_args.items():
        logger.info(f"{k}: {v}")

    # 14. Run training or (separate) test
    if args.mode == 'train':
        logger.info('Starting Training...')
        train(
            continuous=continuous,
            train_env=train_env,
            eval_env=eval_env,
            agent=agent,
            max_episode=args.max_episode,
            max_actions=args.max_actions,
            warmup=args.warmup,
            save_model_dir=args.save_model_dir,
            max_episode_length=args.max_episode_length,
            logger=logger,
            save_per_epochs=args.save_per_epochs,
            eval_interval=args.eval_interval,     # e.g. evaluate every 10 episodes
            eval_episodes=args.eval_episodes     # e.g. 5 episodes each evaluation
        )
        # finish wandb run
        wandb.finish()

    elif args.mode == 'test':
        logger.info('Starting Testing...')
        # You could reuse the 'evaluate' or a separate 'test(...)' function
        from train_test import evaluate
        evaluate(
            agent=agent,
            eval_env=eval_env,
            episodes=args.test_episode,
            max_episode_length=args.max_episode_length,
            logger=logger
        )
    else:
        raise RuntimeError(f'Undefined mode {args.mode}')

if __name__ == "__main__":
    main()
