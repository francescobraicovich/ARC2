import argparse

def init_parser(alg):
    """Initialize argument parser for the specified algorithm."""

    if alg == 'WOLP_DDPG':
        parser = argparse.ArgumentParser(description='WOLP_DDPG')

        # Environment & Training Mode
        parser.add_argument('--env', default='ARC', metavar='ENV', help='Environment to train on')
        parser.add_argument('--mode', default='train', type=str, help='Mode: train/test')
        parser.add_argument('--id', default='0', type=str, help='Experiment ID')
        parser.add_argument('--load', default=False, metavar='L', help='Load a trained model')
        parser.add_argument('--load-model-dir', default='ARC-run780', metavar='LMD', help='Folder to load trained models from')
        parser.add_argument('--eval-interval', default=200, type=int, help='Evaluate model every X episodes')
        parser.add_argument('--eval-episodes', default=2, type=int, help='Number of episodes to evaluate')

        # Episode & Training Settings
        parser.add_argument('--max-episode-length', type=int, default=50, metavar='M', help='Max episode length (default: 50)')  # Changed from 1440
        parser.add_argument('--max-episode', type=int, default=20000, help='Maximum number of episodes')
        parser.add_argument('--max-actions', default=1e8, type=int, help='# max actions')
        parser.add_argument('--test-episode', type=int, default=20, help='Maximum testing episodes')
        parser.add_argument('--warmup', default=200, type=int, help='Time without training but only filling the replay memory')
        parser.add_argument('--bsize', default=3, type=int, help='Minibatch size')
        parser.add_argument('--rmsize', default=100000, type=int, help='Replay memory size')

        # Policy Update Settings
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='Discount factor for rewards (default: 0.99)')
        parser.add_argument('--policy-noise', default=0.1, type=float, help='Noise added to target policy during critic update')
        parser.add_argument('--noise-clip', default=0.25, type=float, help='Range to clip target policy noise')
        parser.add_argument('--policy-delay', default=2, type=int, help='Delay policy updates')
        parser.add_argument('--c-lr', default=3e-4, type=float, help='Critic network learning rate')
        parser.add_argument('--p-lr', default=3e-4, type=float, help='Policy network learning rate (for DDPG)')
        parser.add_argument('--tau-update', default=0.0005, type=float, help='Moving average for target network')
        parser.add_argument('--weight-decay', default=1e-4, type=float, help='L2 Regularization loss weight decay')
        
                            
        # Neural Network Architecture
        parser.add_argument('--hidden1', default=1024, type=int, help='Hidden units in the first fully connected layer')
        parser.add_argument('--hidden2', default=1024, type=int, help='Hidden units in the second fully connected layer')
        parser.add_argument('--actor_critic_type', default='cnn', type=str, help='Type of model (lpn, cnn, mlp)')
        parser.add_argument('--latent_dim', default=48, type=int, help='Latent dimension for encoder')

        # Exploration & Noise
        parser.add_argument('--epsilon', default=100000, type=int, help='Linear decay of exploration policy')
        parser.add_argument('--epsilon_start', default=1.0, type=float, help='Starting epsilon value for resuming training')
        parser.add_argument('--ou_theta', default=0.5, type=float, help='Ornstein-Uhlenbeck noise theta')
        parser.add_argument('--ou_sigma', default=0.2, type=float, help='Ornstein-Uhlenbeck noise sigma')
        parser.add_argument('--ou_mu', default=0.0, type=float, help='Ornstein-Uhlenbeck noise mu')

        # Hardware & GPU Settings
        parser.add_argument('--gpu-ids', type=int, default=[1], nargs='+', help='GPUs to use [-1 for CPU only]')
        parser.add_argument('--gpu-nums', type=int, default=8, help='Number of GPUs to use (default: 1)')

        # Action Embedding & Filtering
        parser.add_argument('--load_action_embedding', default=True, type=bool, help='Load action embedding or not')
        parser.add_argument('--num_experiments_filter', default=120, type=int, help='Number of problems used for filtering actions')
        parser.add_argument('--filter_threshold', default=0.0, type=float, help='Threshold percentage for filtering actions')
        parser.add_argument('--num_experiments_similarity', default=120, type=int, help='Number of problems used for similarity matrix calculation')
        parser.add_argument('--max_embedding', default=1., type=float, help='Maximum value for embedding matrix')
        parser.add_argument('--min_embedding', default=-1., type=float, help='Minimum value for embedding matrix')

        # Miscellaneous
        parser.add_argument('--init_w', default=0.003, type=float, help='Initial weight')
        parser.add_argument('--seed', default=-1, type=int, help='Random seed')
        parser.add_argument('--save_per_epochs', default=25, type=int, help='Save model every X epochs')
        parser.add_argument('--k_neighbors', default=25, type=int, help='Number of neighbors to consider')
        return parser

    else:
        raise RuntimeError(f'Undefined algorithm {alg}')
