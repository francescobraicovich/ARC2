import os
import numpy as np
import torch
import wandb

from utils.util import to_tensor

from dsl.utilities.plot import plot_step

def train(
    continuous,
    train_env,
    eval_env,
    agent,
    max_episode,
    max_actions,
    warmup,
    save_model_dir,
    max_episode_length,
    logger,
    save_per_epochs,
    eval_interval,
    eval_episodes
):
    """
    Train loop that also periodically evaluates on 'eval_env'.

    :param continuous: Whether the action space is continuous (bool)
    :param train_env: Training environment instance
    :param eval_env: Evaluation environment instance
    :param agent: The WolpertingerAgent
    :param max_episode: Max number of training episodes
    :param max_actions: Global maximum number of environment actions
    :param warmup: Steps to fill the buffer with random actions
    :param save_model_dir: Directory to save the model weights
    :param max_episode_length: Max steps per episode
    :param logger: Logger instance
    :param save_per_epochs: Frequency (in episodes) for saving model
    :param eval_interval: Evaluate every N training episodes
    :param eval_episodes: Number of episodes to run in evaluation
    """

    agent.is_training = True

    step = 0
    episode = 0
    episode_steps = 0
    episode_reward = 0.0
    episode_positive_rewards = 0
    num_equal_states = 0
    total_actions_taken = 0

    s_t = None
    positive_rewards_history = []
    rewards_history = []

    while episode < max_episode and total_actions_taken < max_actions:
        if s_t is None:
            state, shape = train_env.reset()
            s_t = to_tensor(state, device=agent.device, requires_grad=True)
            shape = to_tensor(shape, device=agent.device, requires_grad=True)
            agent.reset(s_t, shape)

        # Pick action
        if step <= warmup:
            action, embedded_action = agent.random_action()
        else:
            action, embedded_action = agent.select_action(s_t, shape)

        # Step environment
        (next_state, next_shape), r_t, done, truncated, info = train_env.step(action)
        next_state = to_tensor(next_state, device=agent.device, requires_grad=True)
        next_shape = to_tensor(next_shape, device=agent.device, requires_grad=True)

        total_actions_taken += 1

        # Check for equal states
        if torch.equal(s_t, next_state) and torch.equal(shape, next_shape):
            num_equal_states += 1

        # Count positive reward
        if r_t > 0:
            episode_positive_rewards += 1

        # Check if we hit the max steps in an episode
        if max_episode_length and episode_steps >= max_episode_length - 1:
            truncated = True

        # Observe and update policy
        agent.observe(r_t, next_state, next_shape, done)
        if step > warmup:
            agent.update_policy(step)

        step += 1
        episode_steps += 1
        episode_reward += r_t

        # Move to next state
        s_t = next_state
        shape = next_shape

        # If the episode ends
        if done or truncated or (total_actions_taken >= max_actions):
            episode_reward = float(round(episode_reward, 2))
            logger.info(
                f"[Train] Ep:{episode:<4d} | R:{episode_reward:>7.2f} | "
                f"Steps:{episode_steps:>5d} | EqualStates:{num_equal_states:>5d} "
                f"| PosR:{episode_positive_rewards:>5d} | eps:{agent.epsilon:>6.3f}"
            )

            # wandb logging
            wandb.log({
                "train/episode": episode,
                "train/episode_reward": episode_reward,
                "train/episode_steps": episode_steps,
                "train/positive_rewards": episode_positive_rewards,
                "train/epsilon": agent.epsilon
            })

            # Reset for next episode
            s_t = None
            episode_steps = 0
            episode_reward = 0.0
            episode_positive_rewards = 0
            num_equal_states = 0
            episode += 1

            # --- EVALUATION after eval_interval episodes ---
            if eval_interval > 0 and (episode % eval_interval == 0):
                evaluate(
                    agent=agent,
                    eval_env=eval_env,
                    episodes=eval_episodes,
                    max_episode_length=max_episode_length,
                    logger=logger
                )
                # Resumes training from the same agent state (no reloading needed)

        # Save model periodically
        if step > warmup and episode > 0 and (episode % save_per_epochs == 0):
            agent.save_model(save_model_dir)
            logger.info(f"### Model saved to {save_model_dir} at episode {episode} ###")


def evaluate(
    agent,
    eval_env,
    episodes,
    max_episode_length,
    logger
):
    """
    Runs an evaluation loop using the 'eval_env'. 
    Logs results to wandb under "eval/..." keys.

    :param agent: WolpertingerAgent
    :param eval_env: ARC_Env_Eval environment
    :param episodes: Number of episodes to run for evaluation
    :param max_episode_length: Max steps in an eval episode
    :param logger: Logger
    """
    agent.is_training = False
    agent.eval()

    total_rewards = []

    for ep in range(episodes):
        state, shape = eval_env.reset()
        state = to_tensor(state, device=agent.device, requires_grad=False)
        shape = to_tensor(shape, device=agent.device, requires_grad=False)
        agent.reset(state, shape)

        episode_reward = 0.0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            action, _ = agent.select_action(state, shape, decay_epsilon=False)
            (next_state, next_shape), reward, done, truncated, info = eval_env.step(action)

            next_state = to_tensor(next_state, device=agent.device, requires_grad=False)
            next_shape = to_tensor(next_shape, device=agent.device, requires_grad=False)

            episode_reward += reward
            steps += 1

            state = next_state
            shape = next_shape

            if max_episode_length and steps >= max_episode_length:
                truncated = True

        total_rewards.append(episode_reward)
        logger.info(f"[Eval] Ep:{ep+1}/{episodes} | Reward: {episode_reward:.2f} | Steps: {steps}")

    avg_eval_reward = np.mean(total_rewards)
    logger.info(f"[Eval] Average Reward over {episodes} episodes: {avg_eval_reward:.2f}")

    # wandb logging
    wandb.log({
        "eval/episodes": episodes,
        "eval/average_reward": avg_eval_reward
    })

    # Switch agent back to training mode
    agent.is_training = True
