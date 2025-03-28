from ddpg import DDPG
from utils.util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np

def get_grad_norm(parameters):
    """
    Compute the global 2-norm of gradients for a list of parameters.
    Useful for logging/troubleshooting exploding gradients.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_grad_norm(parameters):
    """
    Compute the global 2-norm of gradients for a list of parameters.
    Useful for logging/troubleshooting exploding gradients.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

class WolpertingerAgent(DDPG):
    """
    WolpertingerAgent extends a base DDPG agent, using the 'Wolpertinger' action selection strategy:
     1) Actor outputs a 'proto-action' in continuous embedding space.
     2) We query the discrete action space to find the k-nearest neighbors to this proto-action.
     3) We evaluate each candidate neighbor with the critic and choose the one with the highest Q-value.

    Key differences from vanilla DDPG:
     - Overridden select_action() to incorporate the Wolpertinger strategy.
     - Overridden random_action() to pick a random embedding, then get an actual discrete action from the action space.
     - Overridden select_target_action() for target Q-value computation using Wolpertinger logic.
    """

    def __init__(self, action_space, nb_states, nb_actions, args, k):
        """
        :param action_space: Custom ARCActionSpace or similar, with .search_point(...)
        :param nb_states: Dimension of states (int)
        :param nb_actions: Dimension of actions (int)
        :param args: Arguments/Hyperparameters (Namespace or dict)
        :param k: Number of neighbors to consider in Wolpertinger
        """

        super().__init__(args, nb_states, nb_actions)

        # Automatically determine the device
        self.device = set_device()
        print(f"[WolpertingerAgent] Using device: {self.device}")

        # Additional parameters
        self.experiment = args.id
        self.action_space = action_space
        self.k_nearest_neighbors = k
        self.MAX_GRAD_NORM = 5.0
        print(f"[WolpertingerAgent] Using {self.k_nearest_neighbors} nearest neighbors.")


        # Move all base DDPG networks to device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)

        # For caching range arrays if using batches
        self.np_aranges = {}
        self.torch_aranges = {}

    def wolp_action(self, x_t, proto_action):
        """
        Given a proto_action (continuous vector from the actor),
        find k nearest discrete actions in the embedding space,
        evaluate them in the critic, and choose the best one.
        """

        # 1) Query the action space for the k-nearest neighbors
        #    distances, indices unused in final selection, but you can log them if desired.
        distances, actions, embedded_actions = self.action_space.search_point(
            proto_action, k=self.k_nearest_neighbors
        )

        # Convert these candidate embedded actions to a tensor on the current device
        embedded_actions = to_tensor(embedded_actions, device=self.device, requires_grad=True)

        # 2) Determine batch size. (Usually 1 for a single state, or B for a minibatch.)
        # TODO: Find batch size from x_t (state)
        batch_size = NotImplementedError

        # Create or reuse pre-cached aranges
        if batch_size not in self.np_aranges:
            self.np_aranges[batch_size] = np.arange(batch_size)
            self.torch_aranges[batch_size] = torch.arange(batch_size, device=self.device)


        # 3) Tile the current states so we can evaluate each candidate with the critic
        x_t_tiled = torch.tile(x_t, (1, 1, 1)) # B, K, embedding_dim
        embedded_actions_tiled = torch.tile(embedded_actions, (1, 1)) # B, K

        # 4) Evaluate Q(s, a) for each candidate. The critic expects (state, shape) tuple + action
        with torch.no_grad():
            q1_values = self.critic1(x_t, embedded_actions_tiled)
            q2_values = self.critic2(x_t, embedded_actions_tiled)
            q_values = torch.min(q1_values, q2_values)

        temperature = 0.1
        q_values = q_values / temperature
        q_probabilities = torch.softmax(q_values, dim=0)

        # 5) Find index of the candidate with maximum Q
        #    If batch_size=1, it's a simple argmax over dimension=0;
        #    Otherwise, we do argmax over dimension=1 if the shape is (k, B, ...)
        if batch_size == 1:
            max_q_indices = torch.argmax(q_values, 0)
            stochastic_index = torch.multinomial(q_probabilities, 1)
        else:
            max_q_indices = torch.argmax(q_values, 1)
            stochastic_index = torch.multinomial(q_probabilities, 1)

        print('\nWolp action:')
        print('Stochastic isdex: ', stochastic_index)
        print('Actions: ', actions)
        selected_action = actions[stochastic_index]

        """
        if batch_size == 1:
            # Single state: Just pick the best index
            selected_action = actions[stochastic_index, :]
            selected_embedded_action = embedded_actions[stochastic_index, :]
        else:
            # If we have a batch, we need to pick the best action for each item in the batch
            reshaped_actions = np.reshape(actions, (self.k_nearest_neighbors, batch_size, -1))
            reshaped_embedded_actions = torch.reshape(
                embedded_actions, (self.k_nearest_neighbors, batch_size, self.nb_actions)
            )
            np_arange = self.np_aranges[batch_size]

            # Convert max_q_indices to a NumPy array for indexing the numpy reshaped_actions
            stochastic_index_np = max_q_indices.cpu().numpy()
            selected_action = reshaped_actions[stochastic_index_np, np_arange, :]
            selected_embedded_action = reshaped_embedded_actions[max_q_indices, self.torch_aranges[batch_size], :]
        """

        return selected_action#, selected_embedded_action

    def select_action(self, x_t, decay_epsilon=True):
        """
        Overridden from DDPG to incorporate Wolpertinger logic.
        1) The base DDPG actor outputs a proto_action (continuous).
        2) We call wolp_action(...) to get the best discrete neighbor.
        """
        # Put networks in eval mode to avoid any training side effects
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

        # Get proto_action (and its embedding) from the DDPG actor
        proto_embedded_action = super().select_action(
            x_t, decay_epsilon=decay_epsilon
        )

        # Evaluate the top-k neighbors in the discrete space
        with torch.no_grad():
            wolp_action = self.wolp_action(x_t, proto_embedded_action)

        # Keep track of the final embedded action used
        self.a_t = wolp_action

        # Switch back to training mode
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        return wolp_action

    def random_action(self):
        """
        Overridden random action:
         1) Sample a random proto_action (continuous) from DDPG.
         2) Query k=1 from the discrete action space and return that single action.
        """
        action = super().random_action()
        #print('Random action from ddpg: ', action)
        #embedded_action = self.action_space.embedding[action]
        #print('Random action embedding: ', embedded_action)
        self.a_t = action
        #print('Returning action: ', action)
        return action

    def update_policy(self, step):

        # ---- Parameter differences for debugging/logging ----
        actor_diff = torch.norm(
            torch.cat([p.view(-1) for p in self.actor.parameters()]) -
            torch.cat([p.view(-1) for p in self.actor_target.parameters()])
        ).item()

        critic1_diff = torch.norm(
            torch.cat([p.view(-1) for p in self.critic1.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic1_target.parameters()])
        ).item()

        critic2_diff = torch.norm(
            torch.cat([p.view(-1) for p in self.critic2.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic2_target.parameters()])
        ).item()

        critics_diff = torch.norm(
            torch.cat([p.view(-1) for p in self.critic1.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic2.parameters()])
        ).item()

        # ---- Sample a batch from memory replay buffer ----
        (state_batch, shape_batch, x_t_batch, action_batch, reward_batch, 
         next_state_batch, next_shape_batch, next_x_t_batch, terminal_batch) = \
            self.memory.sample_and_split(self.batch_size)
        
        action_embedded_batch = self.action_space.embedding[action_batch]
        # TODO: Check why unsqueeze is needed
        #action_batch = torch.unsqueeze(action_batch, 1) # Add back the k-neighrest neighbor dimension

        # ---------------------------------------------------------
        # 1) Compute target actions (with smoothing) using actor_target
        # ---------------------------------------------------------
        with torch.no_grad():
            next_proto_embedded_action_batch = self.actor_target(next_x_t_batch)
            
            # -- TD3: Add clipped noise for smoothing
            """
            noise = (torch.randn_like(proto_embedded_action) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
            proto_embedded_action = proto_embedded_action + noise
            proto_embedded_action = torch.clamp(proto_embedded_action,
                                                self.min_embedding, self.max_embedding)"""
            
            # Wolpertinger: find best discrete neighbor
            #   (proto_embedded_action is a Tensor; if your search_point needs numpy,
            #   convert to numpy, do the search, then come back to Tensor.)
            wolp_action_batch, wolp_embedded_action_batch = self.wolp_action(
                next_x_t_batch, next_proto_embedded_action_batch
            )

            # TODO: Check why unsqueeze is needed
            #wolp_embedded = torch.unsqueeze(wolp_embedded, 1) # Add back the k-neighrest neighbor dimension

            # Evaluate both target critics
            next_q1 = self.critic1_target(next_x_t_batch, wolp_embedded_action_batch)
            next_q2 = self.critic2_target(next_x_t_batch, wolp_embedded_action_batch)

            # TD3: Take the minimum of the two critics for the target
            next_q = torch.min(next_q1, next_q2)

            # Build the target Q-value
            target_q = reward_batch + self.gamma * (1 - terminal_batch.float()) * next_q

        # ---------------------------------------------------------
        # 2) Update both critics
        # ---------------------------------------------------------
        # Critic 1
        self.critic1_optim.zero_grad()
        current_q1 = self.critic1(x_t_batch, action_embedded_batch)
        loss_q1 = F.smooth_l1_loss(current_q1, target_q)
        loss_q1.backward()
        critic1_grad_norm = get_grad_norm(self.critic1.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.MAX_GRAD_NORM)
        self.critic1_optim.step()

        # Critic 2
        self.critic2_optim.zero_grad()
        current_q2 = self.critic2(x_t_batch, action_embedded_batch)
        loss_q2 = F.smooth_l1_loss(current_q2, target_q)
        loss_q2.backward()
        critic2_grad_norm = get_grad_norm(self.critic2.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.MAX_GRAD_NORM)
        self.critic2_optim.step()

        # ---------------------------------------------------------
        # 3) Delayed policy update (actor + target nets)
        # ---------------------------------------------------------
        if step % self.policy_delay == 0:
            # Actor update
            self.actor_optim.zero_grad()
            proto_embedded_action_batch = self.actor(x_t_batch)

            # NOTE: The actor outputs a proto_action, which is a continuous vector.
            #proto_embedded_action_batch = torch.unsqueeze(proto_action_batch, 1) # Add back the k-neighrest neighbor dimension
            
            q_actor = self.critic1(x_t_batch, proto_embedded_action_batch)
            policy_loss = -q_actor.mean()
            policy_loss.backward()
            actor_grad_norm = get_grad_norm(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.MAX_GRAD_NORM)
            self.actor_optim.step()

            # Soft update of targets
            soft_update(self.actor_target, self.actor, self.tau_update)
            soft_update(self.critic1_target, self.critic1, self.tau_update)
            soft_update(self.critic2_target, self.critic2, self.tau_update)
        else:
            policy_loss = torch.tensor(0.)  # no actor update this step
            actor_grad_norm = 0.

        # ---------------------------------------------------------
        # 4) Logging with wandb (optional)
        # ---------------------------------------------------------
        wandb.log({
            "train/loss/critic1_loss": loss_q1.item(),
            "train/loss/critic2_loss": loss_q2.item(),
            "train/loss/actor_loss": policy_loss.item(),
            "train/grad/grad_norm_actor": actor_grad_norm,
            "train/grad/grad_norm_critic1": critic1_grad_norm,
            "train/grad/grad_norm_critic2": critic2_grad_norm,
            "train/diff/actor_diff": actor_diff,
            "train/diff/critic1_diff": critic1_diff,
            "train/diff/critic2_diff": critic2_diff,
            "train/diff/critics_diff": critics_diff
        })
