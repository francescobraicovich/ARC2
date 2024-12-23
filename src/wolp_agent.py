from ddpg import DDPG
import action_space
from util import *
import torch.nn as nn
import torch
criterion = nn.MSELoss()

class WolpertingerAgent(DDPG):
    def __init__(self, nb_states, nb_actions, args, k=5):
        super().__init__(args, nb_states, nb_actions)
        self.experiment = args.id
        
        # according to the papers, it can be scaled to hundreds of millions
        self.action_space = action_space.ARCActionSpace()
        self.k_nearest_neighbors = k

    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def wolp_action(self, s_t, shape, proto_action):
        # get the proto_action's k nearest neighbors
        distances, indices, actions = self.action_space.search_point(proto_action, k=self.k_nearest_neighbors)


        if not isinstance(s_t, np.ndarray):
           s_t = to_numpy(s_t, gpu_used=self.gpu_used)

        # make all the state, action pairs for the critic
        s_t = np.tile(s_t, (self.k_nearest_neighbors, 1, 1, 1, 1))
        shape = np.tile(shape, (self.k_nearest_neighbors, 1, 1, 1))

        #print('Reshaping actions into: ', (self.k_nearest_neighbors,s_t.shape[1], self.nb_actions))
        actions = np.reshape(actions, (self.k_nearest_neighbors,s_t.shape[1], self.nb_actions))
        actions = to_tensor(actions)
        s_t = to_tensor(s_t)
        shape = to_tensor(shape)
        #print('Tiled shape shape', shape.shape)
        
        x = s_t, shape
        
        # evaluate each pair through the critic
        actions_evaluation = torch.squeeze(self.critic(x, actions))


        # find the index of the pair with the maximum value
        max_index = np.argmax(to_numpy(actions_evaluation, gpu_used=self.gpu_used), axis=0)
 
        actions = to_numpy(actions, gpu_used=self.gpu_used)
        selected_action = actions[max_index, np.arange(actions.shape[1]), :]
        if selected_action.shape[0] == 1:
            selected_action = selected_action[0]
        
        #print('Selected action', selected_action)
        # return the best action, i.e., wolpertinger action from the full wolpertinger policy
        return selected_action

    def select_action(self, s_t, shape, decay_epsilon=True):
        # taking a continuous action from the actor
        proto_action = super().select_action(s_t, shape,  decay_epsilon)
        wolp_action = self.wolp_action(s_t, shape, proto_action)
        assert isinstance(wolp_action, np.ndarray)
        self.a_t = wolp_action
        # return the best neighbor of the proto action, this is an action for env step
        return self.a_t

    def random_action(self):
        proto_action = super().random_action()
        distances, indices, actions = self.action_space.search_point(proto_action, 1)
        action = actions[0]
        self.a_t = action
        return action

    def select_target_action(self, s_t, shape):
        x = s_t, shape

        proto_action = self.actor_target(x)
        proto_action = to_numpy(torch.clamp(proto_action, -1.0, 1.0), gpu_used=self.gpu_used)
        action = self.wolp_action(s_t, shape, proto_action)
        return action

    def update_policy(self):
        
        # Sample batch
        state_batch, shape_batch, action_batch, reward_batch, \
        next_state_batch, next_shape_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        # the operation below of critic_target does not require backward_P
        next_state_batch = to_tensor(next_state_batch)
        next_shape_batch = to_tensor(next_shape_batch)
        next_wolp_action_batch = self.select_target_action(next_state_batch, next_shape_batch)
        
        next_states_expaneded = next_state_batch.unsqueeze(0)
        next_shape_expaneded = next_shape_batch.unsqueeze(0)
        next_action_batch_expaneded = to_tensor(next_wolp_action_batch).unsqueeze(0)
        next_state = (next_states_expaneded, next_shape_expaneded)
        
        # next_q_values = Q'(s', wolp_action(s')): calculate the target q value
        next_q_values = self.critic_target(next_state, next_action_batch_expaneded)

        # but it requires bp in computing gradient of critic loss
        next_q_values.volatile = False

        # next_q_values = 0 if is terminal states
        target_q_batch = to_tensor(reward_batch) + self.gamma * to_tensor(terminal_batch) * next_q_values

        # Critic update
        self.critic.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.
        state_batch_unsqueezed = to_tensor(state_batch).unsqueeze(0)
        shape_batch_unsqueezed = to_tensor(shape_batch).unsqueeze(0)
        action_batch_unsqueezed = to_tensor(action_batch).unsqueeze(0)
 
        state_unsqueezed = (state_batch_unsqueezed, shape_batch_unsqueezed)
        q_batch = self.critic(state_unsqueezed, action_batch_unsqueezed)

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()  # computes gradients
        self.critic_optim.step()  # updates the parameters

        # Actor update
        self.actor.zero_grad()

        # self.actor(to_tensor(state_batch)): proto_action_batch

        proto_action_batch = self.actor((to_tensor(state_batch), to_tensor(shape_batch)))
        proto_action_batch_unsqueezed = proto_action_batch.unsqueeze(0)
        policy_loss = -self.critic(state_unsqueezed, proto_action_batch_unsqueezed)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)