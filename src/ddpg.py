import torch
import torch.nn as nn
from torch.optim import Adam
from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from utils.util import *
import random

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

        # Determine the appropriate device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("Using device: {} for ddpg".format(self.device))


        # Set the random seed for reproducibility
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Network configuration for the Actor and Critic networks
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w,
            'type': args.actor_critic_type,
            'latent_dim': args.latent_dim,
            'chunk_size': args.chunk_size
        }

        # Initialize Actor and Critic networks (both primary and target)
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg).to(self.device)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg).to(self.device)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg).to(self.device)
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
        self.shape = None
        self.a_t = None
        self.is_training = True

        self.continious_action_space = False  # Whether the action space is continuous

    def update_policy(self):
        """
        Update the policy by training the Actor and Critic networks.
        """
        pass

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
        Move all networks to the determined device.
        """
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def observe(self, r_t, s_t1, shape1, done):
        """
        Store the most recent transition in the replay buffer.
        """
        if self.is_training:
            assert isinstance(self.a_t, torch.Tensor)
            self.memory.append(self.s_t, self.shape, self.a_t, r_t, done)
            self.s_t = s_t1
            self.shape = shape1

    def random_action(self):
        """
        Generate a random action within the action space bounds.
        """
        action = np.random.uniform(-1., 1., self.nb_actions)
        return action

    def select_action(self, s_t, shape, decay_epsilon=True):
        """
        Select an action based on the current state and exploration policy.

        Args:
            s_t: Current state.
            decay_epsilon: Whether to decay exploration rate after action selection.

        Returns:
            action: Selected action with added exploration noise.
        """
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        if random.random() < self.epsilon:
            action, embedded_action = self.random_action()
            return action, embedded_action
        
        embedded_action = to_numpy(self.actor((s_t.unsqueeze(0), shape.unsqueeze(0))), device=self.device).squeeze(0)
        return None, embedded_action

    def reset(self, s_t, shape):
        """
        Reset the state and noise process for a new episode.
        """
        self.s_t = s_t
        self.shape = shape
        self.random_process.reset_states()

    def seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed: Seed value.
        """
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    def load_weights(self, dir):
        """
        Load the Actor and Critic model weights from the specified directory.

        Args:
            dir: Directory from which to load the model weights.
        """
        if dir is None:
            return

        # Construct paths for the weights
        actor_path = f"../output/{dir}/actor.pt"
        critic_path = f"../output/{dir}/critic.pt"

        # Load Actor model
        if hasattr(self.actor, 'module'):
            # Load into the underlying model for DataParallel
            self.actor.module.load_state_dict(torch.load(actor_path, map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))

        # Load Critic model
        if hasattr(self.critic, 'module'):
            # Load into the underlying model for DataParallel
            self.critic.module.load_state_dict(torch.load(critic_path, map_location=self.device))
        else:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

        print(f"Actor and Critic models loaded from {dir}")


    def save_model(self, output):
        """
        Save the Actor and Critic models to the specified output directory.

        Args:
            output: The directory where the model weights will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output, exist_ok=True)

        # Save the Actor model
        actor_state_dict = (
            self.actor.module.state_dict() if hasattr(self.actor, "module") else self.actor.state_dict()
        )
        torch.save(actor_state_dict, os.path.join(output, "actor.pt"))

        # Save the Critic model
        critic_state_dict = (
            self.critic.module.state_dict() if hasattr(self.critic, "module") else self.critic.state_dict()
        )
        torch.save(critic_state_dict, os.path.join(output, "critic.pt"))

        print(f"Models saved to {output}")
