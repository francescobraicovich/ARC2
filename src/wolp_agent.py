from ddpg import DDPG
from utils.util import *
import torch.nn as nn
import torch
import wandb
import numpy as np

criterion = nn.MSELoss()

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

        print('Using 1-epsilon instead of gamma for target Q-value.')

        # Automatically determine the device
        self.device = set_device()
        print(f"[WolpertingerAgent] Using device: {self.device}")

        # Additional parameters
        self.experiment = args.id
        self.action_space = action_space
        self.k_nearest_neighbors = k
        self.max_embedding = args.max_embedding
        self.min_embedding = args.min_embedding
        print(f"[WolpertingerAgent] Using {self.k_nearest_neighbors} nearest neighbors.")

        # Move all base DDPG networks to device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        # For caching range arrays if using batches
        self.np_aranges = {}
        self.torch_aranges = {}

    def wolp_action(self, s_t, shape, proto_action):
        """
        Given a proto_action (continuous vector from the actor),
        find k nearest discrete actions in the embedding space,
        evaluate them in the critic, and choose the best one.
        """
        # 1) Query the action space for the k-nearest neighbors
        #    distances, indices unused in final selection, but you can log them if desired.
        distances, indices, actions, embedded_actions = self.action_space.search_point(
            proto_action, k=self.k_nearest_neighbors
        )

        # Convert these candidate embedded actions to a tensor on the current device
        embedded_actions = to_tensor(embedded_actions, device=self.device, requires_grad=True)

        # 2) Determine batch size. (Usually 1 for a single state, or B for a minibatch.)
        if len(np.shape(s_t)) == 3:
            # Single state (e.g. shape = (channels, height, width))
            batch_size = 1
        else:
            # e.g. shape = (batch_size, channels, height, width)
            batch_size = np.shape(s_t)[0]

        # Create or reuse pre-cached aranges
        if batch_size not in self.np_aranges:
            self.np_aranges[batch_size] = np.arange(batch_size)
            self.torch_aranges[batch_size] = torch.arange(batch_size, device=self.device)

        # 3) Tile the current states so we can evaluate each candidate with the critic
        s_t_tiled = torch.tile(s_t, (self.k_nearest_neighbors, 1, 1, 1))
        shape_tiled = torch.tile(shape, (self.k_nearest_neighbors, 1, 1))

        # 4) Evaluate Q(s, a) for each candidate. The critic expects (state, shape) tuple + action
        with torch.no_grad():
            q_values = self.critic((s_t_tiled, shape_tiled), embedded_actions)

        # 5) Find index of the candidate with maximum Q
        #    If batch_size=1, it's a simple argmax over dimension=0;
        #    Otherwise, we do argmax over dimension=1 if the shape is (k, B, ...)
        axis = 0 if batch_size == 1 else 1
        max_q_indices = torch.argmax(q_values, dim=axis)

        if batch_size == 1:
            # Single state: Just pick the best index
            selected_action = actions[max_q_indices, :]
            selected_embedded_action = embedded_actions[max_q_indices, :]
        else:
            # If we have a batch, we need to pick the best action for each item in the batch
            reshaped_actions = np.reshape(actions, (self.k_nearest_neighbors, batch_size, -1))
            reshaped_embedded_actions = torch.reshape(
                embedded_actions, (self.k_nearest_neighbors, batch_size, self.nb_actions)
            )
            np_arange = self.np_aranges[batch_size]

            # Convert max_q_indices to a NumPy array for indexing the numpy reshaped_actions
            max_q_indices_np = max_q_indices.cpu().numpy()
            
            selected_action = reshaped_actions[max_q_indices_np, np_arange, :]
            selected_embedded_action = reshaped_embedded_actions[max_q_indices, self.torch_aranges[batch_size], :]


        return selected_action, selected_embedded_action

    def select_action(self, s_t, shape, decay_epsilon=True):
        """
        Overridden from DDPG to incorporate Wolpertinger logic.
        1) The base DDPG actor outputs a proto_action (continuous).
        2) We call wolp_action(...) to get the best discrete neighbor.
        """
        # Put networks in eval mode to avoid any training side effects
        self.actor.eval()
        self.critic.eval()

        # Get proto_action (and its embedding) from the DDPG actor
        proto_action, proto_embedded_action = super().select_action(
            s_t, shape, decay_epsilon=decay_epsilon
        )

        # Evaluate the top-k neighbors in the discrete space
        with torch.no_grad():
            wolp_act, wolp_embedded = self.wolp_action(s_t, shape, proto_embedded_action)

        # Keep track of the final embedded action used
        self.a_t = wolp_embedded

        # Switch back to training mode
        self.actor.train()
        self.critic.train()

        return wolp_act, wolp_embedded

    def random_action(self):
        """
        Overridden random action:
         1) Sample a random proto_action (continuous) from DDPG.
         2) Query k=1 from the discrete action space and return that single action.
        """
        proto_action = super().random_action()  # continuous random
        distances, indices, actions, embedded_actions = self.action_space.search_point(proto_action, 1)
        # We only have 1 neighbor, so select that
        action, embedded_action = actions[0], embedded_actions[0]
        # Convert the embedded action to a tensor on the device
        self.a_t = to_tensor(embedded_action, device=self.device, requires_grad=True)
        return action, embedded_action

    def select_target_action(self, s_t, shape):
        """
        For target Q-value computation:
         1) Get proto_action from the target actor network.
         2) Clamp to [min_embedding, max_embedding].
         3) Use wolp_action to choose best discrete neighbor.
        """
        with torch.no_grad():
            proto_embedded_action = self.actor_target((s_t, shape))
            clamped_action = torch.clamp(
                proto_embedded_action,
                min=self.min_embedding,
                max=self.max_embedding
            )
            # Convert to numpy if the action_space.search_point expects numpy
            clamped_action_np = to_numpy(clamped_action, device=self.device)

            # Evaluate in discrete space
            wolp_act, wolp_embedded = self.wolp_action(s_t, shape, clamped_action_np)

        return wolp_act, wolp_embedded

    def update_policy(self):
        """
        Sample a batch from the replay buffer, compute the target Q,
        update the critic (value function), then update the actor (policy),
        and finally do a soft update of the target networks.
        Also logs relevant metrics to wandb.
        """
        # ---- Parameter differences for debugging/logging ----
        actor_diff = torch.norm(
            torch.cat([p.view(-1) for p in self.actor.parameters()]) -
            torch.cat([p.view(-1) for p in self.actor_target.parameters()])
        ).item()

        critic_diff = torch.norm(
            torch.cat([p.view(-1) for p in self.critic.parameters()]) -
            torch.cat([p.view(-1) for p in self.critic_target.parameters()])
        ).item()

        # ---- Sample a batch from replay buffer ----
        (state_batch, shape_batch, action_batch,
         reward_batch, next_state_batch, next_shape_batch, terminal_batch) = self.memory.sample_and_split(self.batch_size)

        # ---- Compute target Q-value ----
        with torch.no_grad():
            # Next action from target actor + Wolpertinger
            next_act, next_emb_act = self.select_target_action(next_state_batch, next_shape_batch)
            # Evaluate next Q with target critic
            next_q = self.critic_target((next_state_batch, next_shape_batch), next_emb_act)

        target_q = reward_batch + (1-self.epsilon) * (1 - terminal_batch.float()) * next_q

        # ---- Critic update ----
        self.critic_optim.zero_grad()

        q = self.critic((state_batch, shape_batch), action_batch)
        td_error = q - target_q
        value_loss = criterion(q, target_q)

        value_loss.backward()
        actor_grad_norm = get_grad_norm(self.actor.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # ---- Actor update ----
        self.actor_optim.zero_grad()

        proto_action_batch = self.actor((state_batch, shape_batch))  # continuous proto-action
        q_actor = self.critic((state_batch, shape_batch), proto_action_batch)
        policy_loss = -q_actor.mean()

        policy_loss.backward()
        critic_grad_norm = get_grad_norm(self.critic.parameters())
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # ---- Soft update targets ----
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)

        # ---- (Optional) log metrics to wandb ----
        #   * If you want to batch metrics, you can do so in your training loop.
        #   * Otherwise, here is an immediate log of relevant metrics.
        td_mean = td_error.mean().item()
        td_std = td_error.std().item()
        wandb.log({
            "train/critic_loss": value_loss.item(),
            "train/actor_loss": policy_loss.item(),
            "train/td_error_mean": td_mean,
            "train/td_error_std": td_std,
            "train/actor_diff": actor_diff,
            "train/critic_diff": critic_diff,
            # Optionally log gradient norms:
            "train/grad_norm_actor": actor_grad_norm,
            "train/grad_norm_critic": critic_grad_norm
        })
