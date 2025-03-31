import os
import warnings
import logging
import wandb
import torch
import numpy as np

from setproctitle import setproctitle as ptitle
from arg_parser import init_parser

from utils.util import (
    set_device,
    get_output_folder,
    setup_logger
)

from enviroment import ARC_Env
from action_space import ARCActionSpace
from wolp_agent import WolpertingerAgent
from train_test import train, evaluate

from world_model.transformer import EncoderTransformerConfig
from world_model.action_embed import ActionEmbedding
from world_model.state_encode import EncoderTransformer
from world_model.transition_decode import ContextTransformer2D
from world_model.train_world_model import world_model_train

def main():
    warnings.filterwarnings('ignore')

    # 1. Parse arguments
    parser = init_parser('WOLP_DDPG')
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

    # 12. Set up logger
    if args.mode == 'train':
        setup_logger('RS_log', f'{args.save_model_dir}/RS_train_log')
    elif args.mode == 'test':
        setup_logger('RS_log', f'{args.save_model_dir}/RS_test_log')
    else:
        raise RuntimeError(f'Undefined mode {args.mode}')
    logger = logging.getLogger('RS_log')


    args.load_filtered_actions = True
    action_space = ARCActionSpace(args)
    num_filtered_actions = action_space.num_filtered_actions # number of actions after filtering
    
    # Create the world model
    encoder_config = EncoderTransformerConfig(
        emb_dim=args.state_emb_dim,
        latent_dim=args.state_encoded_dim,
        num_heads=args.state_encoder_num_heads,
        num_layers=args.state_encoder_num_layers,
        dropout_rate=args.state_encoder_dropout
    )
    state_encoder = EncoderTransformer(encoder_config)
    action_embedding = ActionEmbedding(
        num_actions=num_filtered_actions,
        embed_dim=args.action_emb_dim,
    )
    transition_decoder = ContextTransformer2D(
        state_encoded_dim=args.state_encoded_dim,
        action_emb_dim=args.action_emb_dim,
        num_layers=args.decoder_num_layers,
        num_heads=args.decoder_num_heads,
        emb_dim=args.decoder_emb_dim,
    )

    if args.load_world_model_weights:
        load_dir = '../output/' + args.load_model_dir
        action_embedding.load_weights(load_dir)
        state_encoder.load_weights(load_dir)
        transition_decoder.load_weights(load_dir)
        logger.info(f"Loaded world model weights from {args.load_model_dir}")

    world_model_args = {
        'epochs': args.world_model_pre_train_epochs,
        'lr': args.world_model_pre_train_lr,
        'batch_size': args.world_model_pre_train_batch_size,
        'max_iter': 20000,
    }

    if args.world_model_pre_train:
        world_model_train(
            state_encoder = state_encoder,
            action_embedder = action_embedding,
            transition_model = transition_decoder,
            world_model_args = world_model_args,
            save_model_dir = args.save_model_dir,
            logger = logger,
            save_per_epochs=10,
        )

    action_embedding = action_embedding.export_weights()
    action_space.load_action_embeddings(action_embedding)
    action_space.create_nearest_neighbors()

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
    continuous = False

    # 9. Create the agent
    agent_args = {
        'nb_actions': num_filtered_actions,
        'args': args,
        'k': args.k_neighbors,
        'action_space': action_space
    }
    agent = WolpertingerAgent(**agent_args)

    # 10. Optionally load model weights
    if args.load:
        agent.load_weights(args.load_model_dir)

    # 13. Log hyperparameters
    d_args = vars(args)

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
            state_encoder=state_encoder,
            agent=agent,
            max_episode=args.max_episode,
            max_actions=args.max_actions,
            warmup=args.warmup,
            save_model_dir=args.save_model_dir,
            max_episode_length=args.max_episode_length,
            logger=logger,
            save_per_epochs=args.save_per_epochs,
            save_memory_at_steps=args.save_memory_at_steps,
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
