from ddpg import DDPG
import action_space
from utils.util import *
import torch.nn as nn
import torch
from utils.util import set_device
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


criterion = nn.MSELoss()
def calculate_gradient_norm(model, print):
    n_params = 0
    sum = 0
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            n_params += 1
            norm = param.grad.norm().item()
            norm = float(norm)
            norms.append(norm)
            sum += norm
        else:
            norms.append(0)
    sum /= n_params if n_params > 0 else 1
    if print:
        print(f'Average gradient norm: {sum:.4f}, max: {max(norms):.4f}, min: {min(norms):.4f}')
    return sum

class WolpertingerAgent(DDPG):
    def __init__(self, action_space, nb_states, nb_actions, args, k):
        super().__init__(args, nb_states, nb_actions)

        # Automatically determine the device
        self.device = set_device()
        print("Using device: {} for Wolpertinger agent".format(self.device))

        self.experiment = args.id
        self.action_space = action_space
        self.k_nearest_neighbors = k
        self.max_embedding = args.max_embedding
        self.min_embedding = args.min_embedding
        print("Using {} nearest neighbors for Wolpertinger agent".format(self.k_nearest_neighbors))

        # Move all networks to the determined device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.n_policy_updates = 0
        self.policy_print_freq = 1
        self.plot_freq = 1000

        self.actor_gradients = []
        self.critic_gradients = []
        self.value_losses = []
        self.policy_losses = []
        self.actor_difference = []
        self.critic_difference = []

        self.np_aranges = {}
        self.torch_aranges = {}
    

    def wolp_action(self, s_t, shape, proto_action):
        # Get the proto_action's k nearest neighbors
        distances, indices, actions, embedded_actions = self.action_space.search_point(proto_action, k=self.k_nearest_neighbors)
        embedded_actions = to_tensor(embedded_actions, device=self.device, requires_grad=True) #NOTE: next step is to move the action space to the device as tensors
        
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
        
        # Evaluate each pair through the critic
        x = s_t, shape
        actions_evaluation = self.critic(x, embedded_actions)

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
        self.actor.train()
        self.critic.train()
        return selected_action, selected_embedded_action

    def select_action(self, s_t, shape, decay_epsilon=True):
        # Take a continuous action from the actor
        self.actor.eval(), self.critic.eval()
        proto_action, proto_embedded_action = super().select_action(s_t, shape, decay_epsilon)
        with torch.no_grad():
            wolp_action, wolp_embedded_action = self.wolp_action(s_t, shape, proto_embedded_action)
        self.a_t = wolp_embedded_action
        self.actor.train(), self.critic.train()
        return wolp_action, wolp_embedded_action

    def random_action(self):
        proto_action = super().random_action()
        distances, indices, actions, embedded_actions = self.action_space.search_point(proto_action, 1)
        action, embedded_action = actions[0], embedded_actions[0]
        self.a_t = to_tensor(embedded_action, device=self.device, requires_grad=True)
        return action, embedded_action

    def select_target_action(self, s_t, shape):
        x = s_t, shape
        proto_embedded_action = self.actor_target(x)
        proto_embedded_action = to_numpy(torch.clamp(proto_embedded_action, self.min_embedding, self.max_embedding), device=self.device)
        action, embedded_action = self.wolp_action(s_t, shape, proto_embedded_action)
        return action, embedded_action
    
    def smooth_array(self, data, window_size=15):
        if len(data) < window_size:
            return data  # Return original if window is too large
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
    def plot_gradients_and_losses(self):
        actor_gradients = np.array(self.actor_gradients)
        critic_gradients = np.array(self.critic_gradients)
        value_losses = np.array(self.value_losses)
        policy_losses = np.array(self.policy_losses)
        actor_differences = np.array(self.actor_difference)
        critic_differences = np.array(self.critic_difference)

        length = len(actor_gradients)
        max_length = 200
        window_size = max(15, length // max_length)
        
        # Apply smoothing
        smooth_actor_gradients = self.smooth_array(actor_gradients, window_size)
        smooth_critic_gradients = self.smooth_array(critic_gradients, window_size)
        smooth_value_losses = self.smooth_array(value_losses, window_size)
        smooth_policy_losses = self.smooth_array(policy_losses, window_size)
        smooth_actor_differences = self.smooth_array(actor_differences, window_size)
        smooth_critic_differences = self.smooth_array(critic_differences, window_size)
        
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        
        
        axs[0, 0].plot(critic_gradients, label='Original', linewidth=0.5)
        axs[0, 0].plot(range(len(smooth_critic_gradients)), smooth_critic_gradients, label='Smoothed', linestyle='solid', linewidth=2, color='red')
        axs[0, 0].set_title('Critic Gradients')
        axs[0, 0].set_yscale('log')
        axs[0, 0].yaxis.set_major_formatter(ScalarFormatter())  # Format y-axis in decimal notation
        axs[0, 0].legend()

        axs[0, 1].plot(actor_gradients, label='Original', linewidth=0.5)
        axs[0, 1].plot(range(len(smooth_actor_gradients)), smooth_actor_gradients, label='Smoothed', linestyle='solid', linewidth=2, color='red')
        axs[0, 1].set_title('Actor Gradients')
        axs[0, 1].set_yscale('log')
        axs[0, 1].yaxis.set_major_formatter(ScalarFormatter())
        axs[0, 1].legend()
        
        axs[1, 0].plot(value_losses, label='Original', linewidth=0.5)
        axs[1, 0].plot(range(len(smooth_value_losses)), smooth_value_losses, label='Smoothed', linestyle='solid', linewidth=2, color='red')
        axs[1, 0].set_title('Value Losses')
        axs[1, 0].set_yscale('log')
        axs[1, 0].yaxis.set_major_formatter(ScalarFormatter())  # Format y-axis in decimal notation
        axs[1, 0].legend()
        
        axs[1, 1].plot(policy_losses, label='Original', linewidth=0.5)
        axs[1, 1].plot(range(len(smooth_policy_losses)), smooth_policy_losses, label='Smoothed', linestyle='solid', linewidth=2, color='red')
        axs[1, 1].set_title('Policy Losses')
        axs[1, 1].set_yscale('log')
        axs[1, 1].yaxis.set_major_formatter(ScalarFormatter())
        axs[1, 1].legend()

        axs[2, 0].plot(critic_differences, label='Original', linewidth=0.5)
        axs[2, 0].plot(range(len(smooth_critic_differences)), smooth_critic_differences, label='Smoothed', linestyle='solid', linewidth=2, color='red')
        axs[2, 0].set_title('Critic Difference with Target')
        axs[2, 0].set_yscale('log')
        axs[2, 0].yaxis.set_major_formatter(ScalarFormatter())
        axs[2, 0].legend()

        axs[2, 1].plot(actor_differences, label='Original', linewidth=0.5)
        axs[2, 1].plot(range(len(smooth_actor_differences)), smooth_actor_differences, label='Smoothed', linestyle='solid', linewidth=2, color='red')
        axs[2, 1].set_title('Actor Difference with Target')
        axs[2, 1].set_yscale('log')
        axs[2, 1].yaxis.set_major_formatter(ScalarFormatter())
        axs[2, 1].legend()

        
        plt.tight_layout()
        plt.show()

    def update_policy(self):

        # Update the policy
        self.n_policy_updates += 1
        PRINT, PLOT = False, False
        if self.n_policy_updates % self.policy_print_freq == 0:
            #PRINT = True
            pass
        if self.n_policy_updates % self.plot_freq == 0:
            PLOT = True
        
        actor_difference = torch.norm(torch.cat([p.view(-1) for p in self.actor.parameters()]) - torch.cat([p.view(-1) for p in self.actor_target.parameters()])).item()
        critic_difference = torch.norm(torch.cat([p.view(-1) for p in self.critic.parameters()]) - torch.cat([p.view(-1) for p in self.critic_target.parameters()])).item()
        self.actor_difference.append(actor_difference), self.critic_difference.append(critic_difference)
        if PRINT:
            print('')
            print('-'*50)
            print('Difference between critic and critic_target: ', critic_difference)
            print('Difference between actor and actor_target: ', actor_difference)

        # Sample batch
        state_batch, shape_batch, action_batch, reward_batch, \
            next_state_batch, next_shape_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        

        with torch.no_grad():
            # Ensure select_target_action does not detach gradients
            next_wolp_action_batch, next_wolp_embedded_action_batch = self.select_target_action(next_state_batch, next_shape_batch)
            next_state = (next_state_batch, next_shape_batch)

            # Compute next Q-values using target critic
            next_q_values = self.critic_target(next_state, next_wolp_embedded_action_batch)
        
        # Compute target Q values
        target_q_batch = reward_batch + self.gamma * (1 - terminal_batch.float()) * next_q_values

        # Critic update
        self.critic_optim.zero_grad()  # Zero gradients before backward
        # 2. Assert gradients are zero after zero_grad()
        for param in self.critic.parameters():
            assert param.grad is None or torch.all(param.grad == 0), "zero_grad() did not properly reset gradients!"

        state = (state_batch, shape_batch)
        q_batch = self.critic(state, action_batch)
        difference = q_batch - target_q_batch
        criterion = torch.nn.MSELoss()  # Define criterion if not defined elsewhere
        value_loss = criterion(q_batch, target_q_batch)
        self.value_losses.append(value_loss.item())
        value_loss.backward()
        
        if PRINT:
            print('')
            print('Critic')
            print(f'Q batch mean: {float(q_batch.mean()):.4f}, std: {float(q_batch.std()):.4f}')
            print(f'Target Q batch mean: {float(target_q_batch.mean()):.4f}, std: {float(target_q_batch.std()):.4f}')
            print(f'Cosine similarity: {float(torch.nn.functional.cosine_similarity(q_batch, target_q_batch, dim=0).mean()):.4f}')
            print(f'Difference mean: {float(difference.mean()):.4f}, std: {float(difference.std()):.4f}')
            print(f'Value loss: {value_loss:.4f}')
        critic_gradient_norm = calculate_gradient_norm(self.critic, PRINT)
        self.critic_gradients.append(critic_gradient_norm)
    
        self.critic_optim.step()

        # Actor update
        self.actor_optim.zero_grad()  # Zero gradients before backward
        # 2. Assert gradients are zero after zero_grad()
        for param in self.actor.parameters():
            assert param.grad is None or torch.all(param.grad == 0), "zero_grad() did not properly reset gradients!"

        proto_action_batch = self.actor((state_batch, shape_batch))  # Ensure correct input format
        q_actor = self.critic(state, proto_action_batch)
        
        policy_loss = -q_actor
        policy_loss_mean = policy_loss.mean()
        self.policy_losses.append(policy_loss_mean.item())
        policy_loss_mean.backward()

        if PRINT:
            print('')
            print('Actor')
            print(f'Proto action mean: {float(proto_action_batch.mean()):.4f}, std: {float(proto_action_batch.std()):.4f}')
            print(f'Proto action max value: {float(proto_action_batch.max()):.4f}, min value: {float(proto_action_batch.min()):.4f}')
            print(f'Policy loss mean: {float(policy_loss.mean()):.4f}, std: {float(policy_loss.std()):.4f}')
            print(f'Polici loss max value: {float(policy_loss.max()):.4f}, min value: {float(policy_loss.min()):.4f}')
        
        actor_gradient_norm = calculate_gradient_norm(self.actor, PRINT)
        self.actor_gradients.append(actor_gradient_norm)
            
        if PRINT:    
            print('-'*50)
            print('')

        if PLOT:
            self.plot_gradients_and_losses()

        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)