import os
import numpy as np
import torch
import wandb
import copy
import logging
import json
from utils.util import to_tensor
from dsl.utilities.plot import plot_step
from collections import deque

def extract_challenge(env, challenge_key, use_test=False, example_index=None, challenges_path='data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json'):
    """
    Extracts a specific challenge from the challenges file and sets up the environment state.
    
    Args:
        env: The ARC environment instance
        challenge_key: The specific challenge key to extract
        use_test: Whether to use the test example (True) or a training example (False)
        example_index: If provided, use this specific training example index instead of random
        challenges_path: Path to the challenges JSON file
        
    Returns:
        tuple: (state, shape, challenge_key) where:
            - state: The initial state of the environment
            - shape: The shape information for the state
            - challenge_key: The key of the selected challenge
    """
    # Load challenges
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    if challenge_key not in challenges:
        raise ValueError(f"Challenge key {challenge_key} not found in challenges file")
    
    # Get the challenge data
    challenge = challenges[challenge_key]
    
    if use_test:
        # Use the test example
        challenge_data = challenge['test']
    else:
        # Use training examples
        challenge_data = challenge['train']
        if example_index is None:
            # Get a random training example if no specific index provided
            example_index = np.random.randint(0, len(challenge_data))
        challenge_data = challenge_data[example_index]
    
    # Get input and output
    input_grid = np.array(challenge_data['input'])
    output_grid = np.array(challenge_data['output'])
    
    # Get shapes
    nrows, ncols = input_grid.shape
    nrows_target, ncols_target = output_grid.shape
    shape = np.array([[nrows, nrows_target], [ncols, ncols_target]])
    
    # Pad the grids
    training_grid = np.zeros(env.observation_shape, dtype=np.int16)
    training_grid[:, :, 0] = env.pad_grid(input_grid)
    training_grid[:, :, 1] = env.pad_grid(output_grid)
    
    # Reset environment state
    env.new_states = deque()
    env.infos = deque()
    env.done = False
    
    # Create initial state and info
    initial_state = (training_grid, shape)
    info = {
        'key': challenge_key,
        'actions': [],
        'action_strings': [],
        'num_actions': 0,
        'solved': False,
        'is_test': use_test,
        'example_index': example_index if not use_test else None
    }
    
    # Update environment state
    env.new_states.append(initial_state)
    env.infos.append(info)
    env.state = env.new_states.popleft()
    env.info = env.infos.popleft()
    
    return env.state, challenge_key

def overfit_train(
    train_env,
    agent,
    challenge_key,
    max_episode,
    max_actions,
    warmup,
    save_model_dir,
    max_episode_length,
    logger,
    save_per_epochs
):
    """
    Fine-tune a pre-trained network on a specific challenge's training examples.
    The agent starts with pre-trained weights and overfits to the training examples
    of the specified challenge to optimize its solution.
    
    Args:
        train_env: Training environment instance
        agent: Pre-trained WolpertingerAgent
        challenge_key: The specific challenge key to overfit to
        max_episode: Max number of training episodes
        max_actions: Global maximum number of environment actions
        warmup: Steps to fill the buffer with random actions
        save_model_dir: Directory to save the model weights
        max_episode_length: Max steps per episode
        logger: Logger instance
        save_per_epochs: Frequency (in episodes) for saving model
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
 

    # Load the challenge to get number of training examples
    with open('data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    num_training_examples = len(challenges[challenge_key]['train'])
    
    # Get the initial state using reset_overfit
    initial_state, initial_shape = reset_overfit(train_env, challenge_key, use_test=False)
    s_t = to_tensor(initial_state, device=agent.device, requires_grad=True)
    shape = to_tensor(initial_shape, device=agent.device, requires_grad=True)
    agent.reset(s_t, shape)
    
    logger.info(f"Starting fine-tuning on challenge key: {challenge_key} with {num_training_examples} training examples")
    logger.info("Using pre-trained weights as starting point")

    while episode < max_episode and total_actions_taken < max_actions:
        if s_t is None:
            # Reset to a random training example using reset_overfit
            state, shape = reset_overfit(train_env, challenge_key, use_test=False)
            s_t = to_tensor(state, device=agent.device, requires_grad=True)
            shape = to_tensor(shape, device=agent.device, requires_grad=True)
            agent.reset(s_t, shape)

        # Pick action with reduced exploration for fine-tuning
        if step <= warmup:
            action, embedded_action = agent.random_action()
        else:
            # Use lower epsilon for fine-tuning to rely more on pre-trained knowledge
            action, embedded_action = agent.select_action(s_t, shape, decay_epsilon=False)

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
                f"[Overfit Train] Ep:{episode:<4d} | R:{episode_reward:>7.2f} | "
                f"Steps:{episode_steps:>5d} | EqualStates:{num_equal_states:>5d} "
                f"| PosR:{episode_positive_rewards:>5d} | eps:{agent.epsilon:>6.3f}"
            )

            # wandb logging
            wandb.log({
                "overfit_train/episode": episode,
                "overfit_train/episode_reward": episode_reward,
                "overfit_train/episode_steps": episode_steps,
                "overfit_train/positive_rewards": episode_positive_rewards,
                "overfit_train/epsilon": agent.epsilon,
                "overfit_train/challenge_key": challenge_key
            })

            # Reset for next episode
            s_t = None
            episode_steps = 0
            episode_reward = 0.0
            episode_positive_rewards = 0
            num_equal_states = 0
            episode += 1

        # Save model periodically
        if step > warmup and episode > 0 and (episode % save_per_epochs == 0):
            agent.save_model(save_model_dir)
            logger.info(f"### Model saved to {save_model_dir} at episode {episode} ###")

    logger.info(f"Training completed on challenge {challenge_key}")

def overfit_evaluate(
    agent,
    eval_env,
    challenge_key,
    episodes,
    max_episode_length,
    logger
):
    """
    Runs an evaluation loop using the 'eval_env' on a specific challenge key.
    Logs results to wandb under "overfit_eval/..." keys.

    :param agent: WolpertingerAgent
    :param eval_env: ARC_Env_Eval environment
    :param challenge_key: The specific challenge key to evaluate on
    :param episodes: Number of episodes to run for evaluation
    :param max_episode_length: Max steps in an eval episode
    :param logger: Logger
    """
    agent.is_training = False
    saved_epsilon = agent.epsilon # Save epsilon value for restoration after evaluation
    agent.epsilon = 0.0 # Set epsilon to 0 for evaluation so that no exploration is performed
    agent.eval()

    total_rewards = []

    for ep in range(episodes):
        state, shape = eval_env.reset()
        while eval_env.info['key'] != challenge_key:
            state, shape = eval_env.reset()
            
        state = to_tensor(state, device=agent.device, requires_grad=False)
        shape = to_tensor(shape, device=agent.device, requires_grad=False)
        agent.reset(state, shape)

        episode_reward = 0.0
        episode_positive_rewards = 0
        num_equal_states = 0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            action, _ = agent.select_action(state, shape, decay_epsilon=False)
            (next_state, next_shape), reward, done, truncated, info = eval_env.step(action)

            next_state = to_tensor(next_state, device=agent.device, requires_grad=False)
            next_shape = to_tensor(next_shape, device=agent.device, requires_grad=False)

            episode_reward += reward
            if reward > 0:
                episode_positive_rewards += 1
            if reward == -3:
                num_equal_states += 1
            if done:
                print("episode solved")
                print('info: ', info)
            steps += 1

            state = next_state
            shape = next_shape

            if max_episode_length and steps >= max_episode_length:
                truncated = True

        total_rewards.append(episode_reward)
        logger.info(f"[Overfit Eval] Ep:{ep+1}/{episodes} | Reward: {episode_reward:.2f} | Steps: {steps} | PosR: {episode_positive_rewards} | EqualStates: {num_equal_states}")

    avg_eval_reward = np.mean(total_rewards)
    logger.info(f"[Overfit Eval] Average Reward over {episodes} episodes: {avg_eval_reward:.2f}")

    # wandb logging
    wandb.log({
        "overfit_eval/episodes": episodes,
        "overfit_eval/average_reward": avg_eval_reward,
        "overfit_eval/challenge_key": challenge_key
    })

    # Switch agent back to training mode
    agent.is_training = True
    agent.epsilon = saved_epsilon # Restore epsilon value

def reset_overfit(env, challenge_key, use_test=False, example_index=None, challenges_path='data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json'):
    """
    Resets the environment to a specific challenge.
    
    Args:
        env: The ARC environment instance
        challenge_key: The specific challenge key to use
        use_test: Whether to use the test example (True) or a training example (False)
        example_index: If provided, use this specific training example index instead of random
        challenges_path: Path to the challenges JSON file
    """
    # Load challenges
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    if challenge_key not in challenges:
        raise ValueError(f"Challenge key {challenge_key} not found in challenges file")
    
    # Get the challenge data
    challenge = challenges[challenge_key]
    
    if use_test:
        # Use the test example
        challenge_data = challenge['test']
    else:
        # Use training examples
        challenge_data = challenge['train']
        if example_index is None:
            # Get a random training example if no specific index provided
            example_index = np.random.randint(0, len(challenge_data))
        challenge_data = challenge_data[example_index]
    
    # Get input and output
    input_grid = np.array(challenge_data['input'])
    output_grid = np.array(challenge_data['output'])
    
    # Get shapes
    nrows, ncols = input_grid.shape
    nrows_target, ncols_target = output_grid.shape
    shape = np.array([[nrows, nrows_target], [ncols, ncols_target]])
    
    # Pad the grids
    training_grid = np.zeros(env.observation_shape, dtype=np.int16)
    training_grid[:, :, 0] = env.pad_grid(input_grid)
    training_grid[:, :, 1] = env.pad_grid(output_grid)
    
    # Reset environment state
    env.new_states = deque()
    env.infos = deque()
    env.done = False
    
    # Create initial state and info
    initial_state = (training_grid, shape)
    info = {
        'key': challenge_key,
        'actions': [],
        'action_strings': [],
        'num_actions': 0,
        'solved': False,
        'is_test': use_test,
        'example_index': example_index if not use_test else None
    }
    
    # Update environment state
    env.new_states.append(initial_state)
    env.infos.append(info)
    env.state = env.new_states.popleft()
    env.info = env.infos.popleft()
    
    return env.state, challenge_key
