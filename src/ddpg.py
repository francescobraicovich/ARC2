import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from utils.util import *
import random
from utils.util import set_device

# Loss function for the Critic network
criterion = nn.MSELoss()
DEVICE = set_device('ddpg.py')

class DDPG(object):
    """
    Deep Deterministic Policy Gradient (DDPG) class.
    Implements the Actor-Critic algorithm for reinforcement learning.
    """
    def __init__(self, args, nb_actions):
        """
        Initialize the DDPG model with the given parameters.

        Args:
            args: Arguments object containing hyperparameters.
            nb_actions: Number of action variables in the environment.
        """

        # Determine the appropriate device
        self.device = DEVICE
        self.save_model_dir = args.save_model_dir

        # Set the random seed for reproducibility
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_actions = nb_actions

        # Hyperparameters for the noise in the policy update
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay

        # Network configuration for the Actor and Critic networks
        net_cfg = {
            'state_encoded_dim': args.state_encoded_dim,
            'action_emb_dim': args.action_emb_dim,
        }

        actor_cfg = {
            **net_cfg,
            'h1_dim': args.h1_dim_actor,
            'h2_dim': args.h2_dim_actor,
                }

        critic_cfg = {
            **net_cfg,
            'h1_dim': args.h1_dim_critic,
            'h2_dim': args.h2_dim_critic,
                }

        # Initialize Actor and Critic networks (both primary and target)
        self.actor = Actor(**actor_cfg).to(self.device)
        self.actor_target = Actor(**actor_cfg).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.p_lr)

        self.critic1 = Critic(**critic_cfg).to(self.device)
        self.critic1_target = Critic(**critic_cfg).to(self.device)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=args.c_lr)

        self.critic2 = Critic(**critic_cfg).to(self.device)
        self.critic2_target = Critic(**critic_cfg).to(self.device)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=args.c_lr)

        # Synchronize target networks with the primary networks
        hard_update(target=self.actor_target, source=self.actor)
        hard_update(target=self.critic1_target, source=self.critic1)
        hard_update(target=self.critic2_target, source=self.critic2)
        
        print('-'*50)
        print('At initialization:')
        print('Difference between actor and actor_target: ', torch.norm(
            torch.cat([p.view(-1) for p in self.actor.parameters()]) -
            torch.cat([p.view(-1) for p in self.actor_target.parameters()])
        ).item())
        print('Difference between critic1 and critic1_target: ', torch.norm(
            torch.cat([p.view(-1) for p in self.critic1.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic1_target.parameters()])
        ).item())
        print('Difference between critic2 and critic2_target: ', torch.norm(
            torch.cat([p.view(-1) for p in self.critic2.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic2_target.parameters()])
        ).item())
        print('Difference between critic1 and critic2: ', torch.norm(
            torch.cat([p.view(-1) for p in self.critic1.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic2.parameters()])
                ).item())
        print('Difference between critic1_target and critic2_target: ', torch.norm(
            torch.cat([p.view(-1) for p in self.critic1_target.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic2_target.parameters()])
                ).item())
        print('-'*50)

        # Initialize replay buffer for experience replay
        self.memory = SequentialMemory(limit=args.rmsize)
        self.memory_size = args.rmsize

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
        self.epsilon = args.epsilon_start

        # Placeholder for the current state and action
        self.x_t = None
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
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def cuda(self):
        """
        Move all networks to the determined device.
        """
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic1.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2.to(self.device)
        self.critic2_target.to(self.device)

    def observe(self, state, shape, x_t, action, r_t, done, num_actions):
        """
        Store the most recent transition in the replay buffer.
        """
        assert type(state) == torch.Tensor, "State should be a torch tensor, got {}".format(type(state))
        assert type(shape) == torch.Tensor, "Shape should be a torch tensor, got {}".format(type(shape))
        assert type(x_t) == torch.Tensor, "x_t should be a torch tensor, got {}".format(type(x_t))
        assert type(action) == torch.Tensor, "Action should be a torch tensor, got {}".format(type(action))
        assert type(r_t) == np.float64, "Reward should be a float, got {}".format(type(r_t))

        # Assume state and shape are torch tensors
        current_state = state[:, :, 0]
        target_state = state[:, :, 1]
        current_shape = shape[0, :]
        target_shape = shape[1, :]

        current_nrows, current_ncols = int(current_shape[0].item()), int(current_shape[1].item())
        target_nrows, target_ncols = int(target_shape[0].item()), int(target_shape[1].item())

        # Extract unpadded parts
        current_unpadded_state = current_state[0:current_nrows, 0:current_ncols]
        target_unpadded_state = target_state[0:target_nrows, 0:target_ncols]

        # Assert no -1 in unpadded regions
        if not torch.sum(current_unpadded_state == -1) == 0:
            print("Current state contains -1 values in unpadded region")
            print('current shape: ', current_shape)
            print('current state: ', current_state)
            assert False, "Current state contains -1 values in unpadded region"
        if not torch.sum(target_unpadded_state == -1) == 0:
            print("Target state contains -1 values in unpadded region")
            print('target shape: ', target_shape)
            print('target state: ', target_state)
            assert False, "Target state contains -1 values in unpadded region"

        # Extract padded parts and check they are -1
        current_padded_mask = torch.ones_like(current_state, dtype=torch.bool)
        current_padded_mask[0:current_nrows, 0:current_ncols] = False
        assert torch.all(current_state[current_padded_mask] == -1), "Current state padded region is not all -1"

        target_padded_mask = torch.ones_like(target_state, dtype=torch.bool)
        target_padded_mask[0:target_nrows, 0:target_ncols] = False
        assert torch.all(target_state[target_padded_mask] == -1), "Target state padded region is not all -1"
        
        if self.is_training:
            self.memory.append(
                observation=state,
                embedded_observation=x_t,
                shape=shape,
                action=action,
                reward=r_t,
                terminal=done,
                num_actions=num_actions
            )

    def random_action(self):
        """
        Generate a random action within the action space bounds.
        """
        action = np.random.randint(self.nb_actions)
        return action # return the index of the action

    def select_action(self, x_t, decay_epsilon=True):
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
            action = self.random_action() # return a random proto-action (See wolp_agent.py)
            embedded_action = self.action_space.embedding[action]
            embedding_std = self.action_space.embedding_std
            gaussian_noise = np.random.normal(0, embedding_std, size=embedded_action.shape)
            proto_embedded_action = embedded_action + gaussian_noise
            return proto_embedded_action
        
        proto_embedded_action = self.actor(x_t) # return the embedded proto-action chosen by the actor
        return proto_embedded_action

    def reset(self, x_t):
        """
        Reset the state and noise process for a new episode.
        """
        self.x_t = x_t
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
        critic1_path = f"../output/{dir}/critic1.pt"
        critic2_path = f"../output/{dir}/critic2.pt"
        actor_target_path = os.path.join(dir, "actor_target.pt")
        critic1_target_path = os.path.join(dir, "critic1_target.pt")
        critic2_target_path = os.path.join(dir, "critic2_target.pt")

        # Always load weights on CPU
        map_location = torch.device("cpu")

        # Load Actor model
        if hasattr(self.actor, 'module'):
            self.actor.module.load_state_dict(torch.load(actor_path, map_location=map_location))
        else:
            self.actor.load_state_dict(torch.load(actor_path, map_location=map_location))
        
        # Load Critic1 model
        if hasattr(self.critic1, 'module'):
            self.critic1.module.load_state_dict(torch.load(critic1_path, map_location=map_location))
        else:
            self.critic1.load_state_dict(torch.load(critic1_path, map_location=map_location))
        
        # Load Critic2 model
        if hasattr(self.critic2, 'module'):
            self.critic2.module.load_state_dict(torch.load(critic2_path, map_location=map_location))
        else:
            self.critic2.load_state_dict(torch.load(critic2_path, map_location=map_location))

        # Load target networks; if target files are missing, fallback to hard_update
        if os.path.exists(actor_target_path):
            if hasattr(self.actor_target, 'module'):
                self.actor_target.module.load_state_dict(torch.load(actor_target_path, map_location=map_location))
            else:
                self.actor_target.load_state_dict(torch.load(actor_target_path, map_location=map_location))
        else:
            hard_update(self.actor_target, self.actor)

        if os.path.exists(critic1_target_path):
            if hasattr(self.critic1_target, 'module'):
                self.critic1_target.module.load_state_dict(torch.load(critic1_target_path, map_location=map_location))
            else:
                self.critic1_target.load_state_dict(torch.load(critic1_target_path, map_location=map_location))
        else:
            hard_update(self.critic1_target, self.critic1)

        if os.path.exists(critic2_target_path):
            if hasattr(self.critic2_target, 'module'):
                self.critic2_target.module.load_state_dict(torch.load(critic2_target_path, map_location=map_location))
            else:
                self.critic2_target.load_state_dict(torch.load(critic2_target_path, map_location=map_location))
        else:
            hard_update(self.critic2_target, self.critic2)

        # Now move all models to self.device
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.actor_target.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)

        print(f"Actor and Critic models and target networks loaded from {dir} and moved to {self.device}")


    def save_model(self, output):
        """
        Save the Actor and Critic models to the specified output directory.

        Args:
            output: The directory where the model weights will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output, exist_ok=True)

        # Save primary models
        actor_state_dict = (
            self.actor.module.state_dict() if hasattr(self.actor, "module") else self.actor.state_dict()
        )
        torch.save(actor_state_dict, os.path.join(output, "actor.pt"))

        critic1_state_dict = (
            self.critic1.module.state_dict() if hasattr(self.critic1, "module") else self.critic1.state_dict()
        )
        torch.save(critic1_state_dict, os.path.join(output, "critic1.pt"))

        critic2_state_dict = (
            self.critic2.module.state_dict() if hasattr(self.critic2, "module") else self.critic2.state_dict()
        )
        torch.save(critic2_state_dict, os.path.join(output, "critic2.pt"))

        # Save target models
        actor_target_state_dict = (
            self.actor_target.module.state_dict() if hasattr(self.actor_target, "module") else self.actor_target.state_dict()
        )
        torch.save(actor_target_state_dict, os.path.join(output, "actor_target.pt"))

        critic1_target_state_dict = (
            self.critic1_target.module.state_dict() if hasattr(self.critic1_target, "module") else self.critic1_target.state_dict()
        )
        torch.save(critic1_target_state_dict, os.path.join(output, "critic1_target.pt"))

        critic2_target_state_dict = (
            self.critic2_target.module.state_dict() if hasattr(self.critic2_target, "module") else self.critic2_target.state_dict()
        )
        torch.save(critic2_target_state_dict, os.path.join(output, "critic2_target.pt"))

        print(f"Models and target networks saved to {output}")

    def save_memory_for_world_model(self):
        """
        Save the replay memory to a file.
        """
        print('Saving memory for world model...')
        print('Save model dir: ', self.save_model_dir)
        self.memory.save_memory_for_world_model_chunk(directory=f"{self.save_model_dir}/memory_chunks")

