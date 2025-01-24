import argparse

def init_parser(alg):

    if alg == 'WOLP_DDPG':

        parser = argparse.ArgumentParser(description='WOLP_DDPG')
        parser.add_argument('--env', default='ARC', metavar='ENV', help='environment to train on')
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
        parser.add_argument('--max-episode-length', type=int, default=50, metavar='M', help='maximum length of an episode (default: 1440)') #NOTE: changed from 1440 to 5
        parser.add_argument('--load', default=False, metavar='L', help='load a trained model')
        parser.add_argument('--load-model-dir', default='ARC-run587', metavar='LMD', help='folder to load trained models from')
        parser.add_argument('--gpu-ids', type=int, default=[1], nargs='+', help='GPUs to use [-1 CPU only]')
        parser.add_argument('--gpu-nums', type=int, default=8, help='#GPUs to use (default: 1)')
        parser.add_argument('--max-episode', type=int, default=20000, help='maximum #episode.')
        parser.add_argument('--test-episode', type=int, default=100, help='maximum testing #episode.')
        parser.add_argument('--max-actions', default=200000, type=int, help='# max actions')
        parser.add_argument('--id', default='0', type=str, help='experiment id')
        parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
        parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connect layer')
        parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
        parser.add_argument('--c-lr', default=0.01, type=float, help='critic net learning rate')
        parser.add_argument('--p-lr', default=0.01, type=float, help='policy net learning rate (only for DDPG)')
        parser.add_argument('--warmup', default=256, type=int, help='time without training but only filling the replay memory')
        parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
        parser.add_argument('--rmsize', default=25000, type=int, help='memory size')
        parser.add_argument('--window_length', default=1, type=int, help='')
        parser.add_argument('--tau-update', default=0.05, type=float, help='moving average for target network')
        parser.add_argument('--ou_theta', default=0.5, type=float, help='noise theta')
        parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
        parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
        parser.add_argument('--max_episode_length', default=500, type=int, help='')
        parser.add_argument('--init_w', default=0.003, type=float, help='')
        parser.add_argument('--epsilon', default=300000, type=int, help='Linear decay of exploration policy')
        parser.add_argument('--seed', default=-1, type=int, help='')
        parser.add_argument('--weight-decay', default=0.00001, type=float, help='weight decay for L2 Regularization loss')
        parser.add_argument('--save_per_epochs', default=50, type=int, help='save model every X epochs')
        parser.add_argument('--actor_critic_type', default='cnn', type=str, help='type of model to use (lpn, cnn, mlp)')
        parser.add_argument('--k_neighbors', default=100, type=int, help='number of neighbors to consider')
        parser.add_argument('--load_action_embedding', default=True, type=bool, help='load action embedding or not')
        parser.add_argument('--latent_dim', default=48, type=int, help='latent dimension for encoder')
        parser.add_argument('--chunk_size', default=10, type=int, help='chunk size for training encoder')
        parser.add_argument('--epsilon_start', default=1.0, type=float, help='starting epsilon value, useful for resuming training')
        return parser
    else:
        raise RuntimeError('undefined algorithm {}'.format(alg))