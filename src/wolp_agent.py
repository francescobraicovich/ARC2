from ddpg import DDPG
import action_space
from util import *
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

    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def wolp_action(self, s_t, shape, proto_action):
        # Get the proto_action's k nearest neighbors
        distances, indices, actions = self.action_space.search_point(proto_action, k=self.k_nearest_neighbors)
        actions = to_tensor(actions, device=self.device) #NOTE: next step is to move the action space to the device as tensors

        # Make all the state, action pairs for the critic        
        s_t = torch.tile(s_t, (self.k_nearest_neighbors, 1, 1, 1, 1))
        shape = torch.tile(shape, (self.k_nearest_neighbors, 1, 1, 1))

        actions = torch.reshape(actions, (self.k_nearest_neighbors, s_t.shape[1], self.nb_actions))

        #actions = to_tensor(actions, device=self.device)
        x = s_t, shape

        # Evaluate each pair through the critic
        actions_evaluation = torch.squeeze(self.critic(x, actions))

        # Find the index of the pair with the maximum value
        max_index = torch.argmax(actions_evaluation, dim=0)

        # Select the actions based on the maximum index
        selected_action = actions[max_index, torch.arange(actions.size(1)), :]

        # Adjust shape if necessary
        if selected_action.size(0) == 1:
            selected_action = selected_action[0]
        selected_action = to_numpy(selected_action, device=self.device)

        return selected_action

    def select_action(self, s_t, shape, decay_epsilon=True):
        # Take a continuous action from the actor
        proto_action = super().select_action(s_t, shape, decay_epsilon)
    
        wolp_action = self.wolp_action(s_t, shape, proto_action)
        assert isinstance(wolp_action, np.ndarray)
        self.a_t = to_tensor(wolp_action, device=self.device)
        return wolp_action

    def random_action(self):
        proto_action = super().random_action()
        distances, indices, actions = self.action_space.search_point(proto_action, 1)
        action = actions[0]
        self.a_t = to_tensor(action, device=self.device)
        return action

    def select_target_action(self, s_t, shape):
        x = s_t, shape

        proto_action = self.actor_target(x)
        proto_action = to_numpy(torch.clamp(proto_action, -1.0, 1.0), device=self.device)
        action = self.wolp_action(s_t, shape, proto_action)
        return action

    def update_policy(self):
        # Sample batch
        state_batch, shape_batch, action_batch, reward_batch, \
            next_state_batch, next_shape_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        
        # Prepare for the target q batch
        next_wolp_action_batch = self.select_target_action(next_state_batch, next_shape_batch)

        next_states_expanded = next_state_batch.unsqueeze(0)
        next_shape_expanded = next_shape_batch.unsqueeze(0)
        next_action_batch_expanded = to_tensor(next_wolp_action_batch, device=self.device).unsqueeze(0)
        next_state = (next_states_expanded, next_shape_expanded)

        # Next Q values
        next_q_values = self.critic_target(next_state, next_action_batch_expanded)

        # Handle terminal states
        target_q_batch = reward_batch + self.gamma * terminal_batch * next_q_values

        # Critic update
        self.critic.zero_grad()
        state_batch_unsqueezed = state_batch.unsqueeze(0)
        shape_batch_unsqueezed = shape_batch.unsqueeze(0)
        action_batch_unsqueezed = action_batch.unsqueeze(0)

        state_unsqueezed = (state_batch_unsqueezed, shape_batch_unsqueezed)
        q_batch = self.critic(state_unsqueezed, action_batch_unsqueezed)

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        proto_action_batch = self.actor((state_batch, shape_batch))
        proto_action_batch_unsqueezed = proto_action_batch.unsqueeze(0)
        policy_loss = -self.critic(state_unsqueezed, proto_action_batch_unsqueezed)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)