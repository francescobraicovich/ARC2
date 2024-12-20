import torch
import torch.nn as nn
from torch.optim import Adam

# Importing custom modules for Actor-Critic models, memory management, and utilities
from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# Loss function for the Critic network
criterion = nn.MSELoss()

class DDPG(object):
    """
    Deep Deterministic Policy Gradient (DDPG) class.
    Implements the Actor-Critic algorithm for reinforcement learning.
    """
    def __init__(self, args, nb_states, nb_actions):
        """
        Initialize the DDPG model with the given parameters.

        Args:
            args: Arguments object containing hyperparameters.
            nb_states: Number of state variables in the environment.
            nb_actions: Number of action variables in the environment.
        """
        USE_CUDA = torch.cuda.is_available()

        # Set the random seed for reproducibility
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Determine GPU usage based on availability and user settings
        self.gpu_ids = [i for i in range(args.gpu_nums)] if USE_CUDA and args.gpu_nums > 0 else [-1]
        self.gpu_used = True if self.gpu_ids[0] >= 0 else False
        self.device = torch.device('cuda:0' if self.gpu_used else 'cpu')


        # Network configuration for the Actor and Critic networks
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }

        # Initialize Actor and Critic networks (both primary and target)
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.c_lr, weight_decay=args.weight_decay)

        # Synchronize target networks with the primary networks
        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        # Initialize replay buffer for experience replay
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Initialize Ornstein-Uhlenbeck process for action exploration noise
        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.nb_actions,
            theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma
        )

        # Set hyperparameters
        self.batch_size = args.bsize
        self.tau_update = args.tau_update
        self.gamma = args.gamma

        # Exploration rate decay and initial exploration rate
        self.depsilon = 1.0 / args.epsilon
        self.epsilon = 1.0

        # Placeholder for the current state and action
        self.s_t = None
        self.a_t = None
        self.is_training = True

        self.continious_action_space = False  # Whether the action space is continuous

    def update_policy(self):
        """
        Update the policy by training the Actor and Critic networks.
        """
        pass

    def cuda_convert(self):
        """
        Convert the model to GPU(s) if available and specified.
        """
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] >= 0:
            with torch.cuda.device(self.gpu_ids[0]):
                print('Model converted to CUDA.')
                self.cuda()
        elif len(self.gpu_ids) > 1:
            self.data_parallel()
            self.cuda()
            self.to_device()
            print('Model converted to CUDA and parallelized.')

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        """
        Move all networks to GPU memory.
        """
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def data_parallel(self):
        """
        Enable data parallelism for the model across multiple GPUs.
        """
        self.actor = nn.DataParallel(self.actor, device_ids=self.gpu_ids)
        self.actor_target = nn.DataParallel(self.actor_target, device_ids=self.gpu_ids)
        self.critic = nn.DataParallel(self.critic, device_ids=self.gpu_ids)
        self.critic_target = nn.DataParallel(self.critic_target, device_ids=self.gpu_ids)

    def to_device(self):
        """
        Move all networks to the primary GPU device.
        """
        device = torch.device(f'cuda:{self.gpu_ids[0]}')
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

    def observe(self, r_t, s_t1, done):
        """
        Store the most recent transition in the replay buffer.
        """
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        """
        Generate a random action within the action space bounds.
        """
        action = np.random.uniform(-1., 1., self.nb_actions)
        return action

    def select_action(self, s_t, decay_epsilon=True):
        """
        Select an action based on the current state and exploration policy.

        Args:
            s_t: Current state.
            decay_epsilon: Whether to decay exploration rate after action selection.

        Returns:
            action: Selected action with added exploration noise.
        """
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t]), gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])),
            gpu_used=self.gpu_used
        ).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        return action

    def reset(self, s_t):
        """
        Reset the state and noise process for a new episode.
        """
        self.s_t = s_t
        self.random_process.reset_states()

    def load_weights(self, dir):
        """
        Load model weights from the specified directory.

        Args:
            dir: Directory containing the saved weights.
        """
        if dir is None:
            return

        map_location = lambda storage, loc: storage.cuda(self.gpu_ids) if self.gpu_used else storage
        self.actor.load_state_dict(torch.load(f'output/{dir}/actor.pkl', map_location=map_location))
        self.critic.load_state_dict(torch.load(f'output/{dir}/critic.pkl', map_location=map_location))
        print('Model weights loaded.')

    def save_model(self, output):
        """
        Save the model weights to the specified directory.

        Args:
            output: Directory to save the weights.
        """
        save_actor = f'{output}/actor.pt'
        save_critic = f'{output}/critic.pt'

        if len(self.gpu_ids) == 1 and self.gpu_ids[0] > 0:
            with torch.cuda.device(self.gpu_ids[0]):
                torch.save(self.actor.state_dict(), save_actor)
                torch.save(self.critic.state_dict(), save_critic)
        elif len(self.gpu_ids) > 1:
            torch.save(self.actor.module.state_dict(), save_actor)
            torch.save(self.critic.module.state_dict(), save_critic)
        else:
            torch.save(self.actor.state_dict(), save_actor)
            torch.save(self.critic.state_dict(), save_critic)

    def seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed: Seed value.
        """
        torch.manual_seed(seed)
        if len(self.gpu_ids) > 0:
            torch.cuda.manual_seed_all(seed)
