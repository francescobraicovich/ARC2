import argparse

PRESETS = {
    'generate_world_model_data': {
        'save_memory_at_steps': 2*int(1e4),
        'max_episode_length': 75,
    },
    # Add more presets here
}
def check_presets(PRESETS):
    world_model_preset = PRESETS['generate_world_model_data']

    world_model_preset['max_actions'] = world_model_preset['save_memory_at_steps'] + 1
    world_model_preset['max_episode'] = world_model_preset['save_memory_at_steps'] + 1
    world_model_preset['eval_interval'] = world_model_preset['save_memory_at_steps'] + 1
    world_model_preset['save_per_epochs'] = world_model_preset['save_memory_at_steps'] + 1
    world_model_preset['warmup'] = world_model_preset['save_memory_at_steps'] + 1
    world_model_preset['eval_episodes'] = 0
    world_model_preset['world_model_pre_train'] = False
    world_model_preset['load_world_model_weights'] = False

    world_model_preset['state_encoded_dim'] = 2
    world_model_preset['action_emb_dim'] = 2
    world_model_preset['state_emb_dim'] = 2
    world_model_preset['state_encoder_num_heads'] = 1
    world_model_preset['state_encoder_num_layers'] = 1
    world_model_preset['state_encoder_dropout'] = 0

    # set the actor-critic architecture to low values
    world_model_preset['h1_dim_actor'] = 2
    world_model_preset['h2_dim_actor'] = 2
    world_model_preset['h1_dim_critic'] = 2
    world_model_preset['h2_dim_critic'] = 2

    # assert the save_memory_at_steps is divisible by 50
    assert world_model_preset['save_memory_at_steps'] % 50 == 0, f"save_memory_at_steps should be divisible by 50, but got {world_model_preset['save_memory_at_steps']}"
    if world_model_preset['save_memory_at_steps'] < 2501:
            world_model_preset['rmsize'] = world_model_preset['save_memory_at_steps'] + 1
    elif world_model_preset['save_memory_at_steps'] < 10001:
         # assert is divisible by 500 for the interval
        assert world_model_preset['save_memory_at_steps'] % 500 == 0, f"save_memory_at_steps should be divisible by 500 in the interval (2500, 10000), but got {world_model_preset['save_memory_at_steps']}"
        world_model_preset['rmsize'] = 2500
    elif world_model_preset['save_memory_at_steps'] < 50001:
         # assert is divisible by 5000 for the interval
        assert world_model_preset['save_memory_at_steps'] % 5000 == 0, f"save_memory_at_steps should be divisible by 5000 in the interval (10000, 50000), but got {world_model_preset['save_memory_at_steps']}"
        world_model_preset['rmsize'] = 5000
    elif world_model_preset['save_memory_at_steps'] < 100001:
            # assert is divisible by 10000 for the interval
            assert world_model_preset['save_memory_at_steps'] % 10000 == 0, f"save_memory_at_steps should be divisible by 10000 in the interval (50000, 100000), but got {world_model_preset['save_memory_at_steps']}"
            world_model_preset['rmsize'] = 10000
    else:
        # assert is divisible by 20000 for the interval
        assert world_model_preset['save_memory_at_steps'] % 20000 == 0, f"save_memory_at_steps should be divisible by 20000 in the interval (100000, inf), but got {world_model_preset['save_memory_at_steps']}"
        world_model_preset['rmsize'] = 20000


    print('World model preset:', world_model_preset)

    # assert the embedding dimension is divisible by the number of heads
    assert world_model_preset['state_emb_dim'] % world_model_preset['state_encoder_num_heads'] == 0
    
    return PRESETS

PRESETS = check_presets(PRESETS)

def init_parser(alg):
    """Initialize argument parser for the specified algorithm."""

    if alg == 'WOLP_DDPG':
        parser = argparse.ArgumentParser(description='WOLP_DDPG')

        # PRESETS
        parser.add_argument('--generate_world_model_data', default=False, type=bool, help='Generate world model data, overwrites some parameters')

        # Environment & Training Mode
        parser.add_argument('--env', default='ARC', metavar='ENV', help='Environment to train on')
        parser.add_argument('--mode', default='train', type=str, help='Mode: train/test')
        parser.add_argument('--id', default='0', type=str, help='Experiment ID')
        parser.add_argument('--load', default=False, metavar='L', help='Load a trained model')
        parser.add_argument('--load-model-dir', default='ARC-run19', metavar='LMD', help='Folder to load trained models from')
        parser.add_argument('--eval-interval', default=200, type=int, help='Evaluate model every X episodes')
        parser.add_argument('--eval-episodes', default=25, type=int, help='Number of episodes to evaluate')
        parser.add_argument('--plot_interval', default=5,type=int, help='Plot every X epochs')

        # Episode & Training Settings
        parser.add_argument('--max-episode-length', type=int, default=30, metavar='M', help='Max episode length (default: 50)')
        parser.add_argument('--max-episode', type=int, default=500000, help='Maximum number of episodes')
        parser.add_argument('--max-actions', default=1e9, type=int, help='# max actions')
        parser.add_argument('--test-episode', type=int, default=20, help='Maximum testing episodes')
        parser.add_argument('--warmup', default=190, type=int, help='Time without training but only filling the replay memory')
        parser.add_argument('--bsize', default=32, type=int, help='Minibatch size')
        parser.add_argument('--rmsize', default=100000, type=int, help='Replay memory size')

        # Policy Update Settings
        parser.add_argument('--gamma', type=float, default=0.2, metavar='G', help='Discount factor for rewards (default: 0.99)')
        parser.add_argument('--policy-noise', default=0, type=float, help='Noise added to target policy during critic update')
        parser.add_argument('--noise-clip', default=0.25, type=float, help='Range to clip target policy noise')
        parser.add_argument('--policy-delay', default=2, type=int, help='Delay policy updates')
        parser.add_argument('--c-lr', default=1e-4, type=float, help='Critic network learning rate')
        parser.add_argument('--p-lr', default=1e-4, type=float, help='Policy network learning rate (for DDPG)')
        parser.add_argument('--tau-update', default=0.0005, type=float, help='Moving average for target network')
        parser.add_argument('--weight-decay', default=1e-4, type=float, help='L2 Regularization loss weight decay')
                            
        # Actor-Critic Architecture
        parser.add_argument('--h1_dim_actor', default=2048, type=int, help='Hidden units in the first fully connected layer')
        parser.add_argument('--h2_dim_actor', default=1024, type=int, help='Hidden units in the second fully connected layer')
        parser.add_argument('--h1_dim_critic', default=2048, type=int, help='Hidden units in the first fully connected layer')
        parser.add_argument('--h2_dim_critic', default=1024, type=int, help='Hidden units in the second fully connected layer')

        # World Model Embedding
        parser.add_argument('--world_model_pre_train', default=True, type=bool, help='Pre-train world model before the RL loop')
        parser.add_argument('--load-memory-dir', default='ARC-run5', metavar='LMD', help='Folder to load memory from')
        parser.add_argument('--load_world_model_weights', default=False, type=bool, help='Load pre-trained world model from load-model-dir folder')
        parser.add_argument('--world_model_pre_train_epochs', default=100, type=int, help='Number of epochs for pre-training world model')
        parser.add_argument('--world_model_pre_train_batch_size', default=32, type=int, help='Batch size for pre-training world model')
        parser.add_argument('--world_model_pre_train_lr', default=1e-4, type=float, help='Learning rate for pre-training world model')
    
        parser.add_argument('--state_encoded_dim', default=512, type=int, help='State latent (encoded) dimension')
        parser.add_argument('--action_emb_dim', default=256, type=int, help='Action embedding dimension')
        parser.add_argument('--state_emb_dim', default=196, type=int, help='Embedding dimension for state representation in attention')
        parser.add_argument('--state_encoder_num_heads', default=4, type=int, help='Number of attention heads in state encoder')
        parser.add_argument('--state_encoder_num_layers', default=3, type=int, help='Number of transformer layers in state encoder')
        parser.add_argument('--state_encoder_dropout', default=0.1, type=float, help='Dropout rate in state encoder')
        parser.add_argument('--decoder_emb_dim', default=128, type=int, help='Embedding dimension for decoder')
        parser.add_argument('--decoder_num_heads', default=4, type=int, help='Number of attention heads in decoder')
        parser.add_argument('--decoder_num_layers', default=3, type=int, help='Number of transformer layers in decoder')

        # Exploration & Noise
        parser.add_argument('--epsilon', default=50000, type=int, help='Linear decay of exploration policy')
        parser.add_argument('--epsilon_start', default=1.0, type=float, help='Starting epsilon value for resuming training')
        parser.add_argument('--ou_theta', default=0.5, type=float, help='Ornstein-Uhlenbeck noise theta')
        parser.add_argument('--ou_sigma', default=0.2, type=float, help='Ornstein-Uhlenbeck noise sigma')
        parser.add_argument('--ou_mu', default=0.0, type=float, help='Ornstein-Uhlenbeck noise mu')

        # Hardware & GPU Settings
        parser.add_argument('--gpu-ids', type=int, default=[1], nargs='+', help='GPUs to use [-1 for CPU only]')
        parser.add_argument('--gpu-nums', type=int, default=8, help='Number of GPUs to use (default: 1)')

        # Filtering Action Space
        parser.add_argument('--load_filtered_actions', default=True, type=bool, help='Load filtered actions or not')
        parser.add_argument('--num_experiments_filter', default=120, type=int, help='Number of problems used for filtering actions')
        parser.add_argument('--filter_threshold', default=0.3, type=float, help='Threshold percentage for filtering actions')

        # Miscellaneous
        parser.add_argument('--init_w', default=0.003, type=float, help='Initial weight')
        parser.add_argument('--seed', default=-1, type=int, help='Random seed')
        parser.add_argument('--save_per_epochs', default=200, type=int, help='Save model every X epochs')
        parser.add_argument('--save_memory_at_steps', default=False, type=int, help='Save memory every X epochs')
        parser.add_argument('--memory_chunk_size', default=False, type=int, help='Chunk size for memory')
        parser.add_argument('--k_neighbors', default=150, type=int, help='Number of neighbors to consider')
        parser.add_argument('--wandb_log', default=True, type=bool, help='Log to wandb')
        
        # Apply preset defaults if provided
        if parser.parse_args().generate_world_model_data:
            print('Generating world model data, overwriting some parameters')
            parser.set_defaults(**PRESETS['generate_world_model_data'])
        
        return parser
    else:
        raise RuntimeError(f'Undefined algorithm {alg}')



