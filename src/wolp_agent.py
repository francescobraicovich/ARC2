from ddpg import DDPG
import action_space
from utils.util import *
import torch.nn as nn
import torch
criterion = nn.MSELoss()

class WolpertingerAgent(DDPG):
    def __init__(self, nb_states, nb_actions, args, k):
        super().__init__(args, nb_states, nb_actions)

        # Automatically determine the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("Using device: {} for Wolpertinger agent".format(self.device))

        self.experiment = args.id
        self.action_space = action_space.ARCActionSpace()
        self.k_nearest_neighbors = k
        print("Using {} nearest neighbors for Wolpertinger agent".format(self.k_nearest_neighbors))

        # Move all networks to the determined device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.np_aranges = {}
        self.torch_aranges = {}

    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space
    

    def wolp_action(self, s_t, shape, proto_action):
        
        # Get the proto_action's k nearest neighbors
        distances, indices, actions, embedded_actions = self.action_space.search_point(proto_action, k=self.k_nearest_neighbors)
        embedded_actions = to_tensor(embedded_actions, device=self.device) #NOTE: next step is to move the action space to the device as tensors
        
        # Get the batch size and create the aranges if necessary for later indexing
        if len(np.shape(s_t)) == 3:
            batch_size = 1
        else:
            batch_size = np.shape(s_t)[0]
        if batch_size not in self.np_aranges:
            self.np_aranges[batch_size] = np.arange(batch_size)
            self.torch_aranges[batch_size] = torch.arange(batch_size, device=self.device)
        
        # Make all the state, action pairs for the critic        
        s_t = torch.tile(s_t, (self.k_nearest_neighbors, 1, 1, 1))
        shape = torch.tile(shape, (self.k_nearest_neighbors, 1, 1))

        # Reshape the actions and embedded actions
        #embedded_actions = torch.reshape(embedded_actions, (self.k_nearest_neighbors, batch_size, self.nb_actions))
        #actions = np.reshape(actions, (self.k_nearest_neighbors, batch_size, 3))

        # Evaluate each pair through the critic
        x = s_t, shape
        actions_evaluation = torch.squeeze(self.critic(x, embedded_actions))

        # Find the index of the pair with the maximum value
        axis = 0 if batch_size == 1 else 1
        torch_max_index = torch.argmax(actions_evaluation, dim=axis)
        np_max_index = torch_max_index if batch_size == 1 else to_numpy(torch_max_index, device=self.device)
        
        if batch_size == 1:
            selected_action = actions[np_max_index, :]
            selected_embedded_action = embedded_actions[torch_max_index, :]
        else:
            # Select the actions based on the maximum indes
            np_arange = self.np_aranges[batch_size]
            torch_arange = self.torch_aranges[batch_size]

            # Reshape the actions and embedded actions
            reshaped_actions = np.reshape(actions, (self.k_nearest_neighbors, batch_size, 3))
            reshaped_embedded_actions = torch.reshape(embedded_actions, (self.k_nearest_neighbors, batch_size, self.nb_actions))
            
            selected_action = reshaped_actions[np_max_index, np_arange, :]
            selected_embedded_action = reshaped_embedded_actions[torch_max_index, torch_arange, :]

        return selected_action, selected_embedded_action

    def select_action(self, s_t, shape, decay_epsilon=True):
        # Take a continuous action from the actor
        proto_action, proto_embedded_action = super().select_action(s_t, shape, decay_epsilon)
        wolp_action, wolp_embedded_action = self.wolp_action(s_t, shape, proto_embedded_action)
        self.a_t = wolp_embedded_action
        return wolp_action, wolp_embedded_action

    def random_action(self):
        proto_action = super().random_action()
        distances, indices, actions, embedded_actions = self.action_space.search_point(proto_action, 1)
        action, embedded_action = actions[0], embedded_actions[0]
        self.a_t = to_tensor(embedded_action, device=self.device)
        return action, embedded_action

    def select_target_action(self, s_t, shape):
        x = s_t, shape
        proto_embedded_action = self.actor_target(x)
        proto_embedded_action = to_numpy(torch.clamp(proto_embedded_action, -1.0, 1.0), device=self.device)
        action, embedded_action = self.wolp_action(s_t, shape, proto_embedded_action)
        return action, embedded_action

    def update_policy(self):
        # Sample batch
        state_batch, shape_batch, action_batch, reward_batch, \
            next_state_batch, next_shape_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        
        # Prepare for the target q batch
        next_wolp_action_batch, next_wolp_embedded_action_batch = self.select_target_action(next_state_batch, next_shape_batch)
        next_state = (next_state_batch, next_shape_batch)

        # Next Q values
        next_q_values = self.critic_target(next_state, next_wolp_embedded_action_batch)

        # Handle terminal states
        target_q_batch = reward_batch + self.gamma * terminal_batch * next_q_values

        # Critic update
        self.critic.zero_grad()

        state = (state_batch, shape_batch)
        q_batch = self.critic(state, action_batch)

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        proto_action_batch = self.actor((state_batch, shape_batch))
        proto_action_batch_unsqueezed = proto_action_batch.unsqueeze(0)
        policy_loss = -self.critic(state, proto_action_batch_unsqueezed)

        policy_loss = policy_loss.mean()
        print('Policy loss after mean: {}'.format(policy_loss))
        policy_loss.backward()

        # Check gradients
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"Gradients for {name}: {param.grad.norm().item()}")  # Print gradient norm
            else:
                print(f"Gradients for {name}: None")

        self.actor_optim.step()
        
        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)