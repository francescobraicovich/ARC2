import os
import numpy as np
import torch
import wandb
import copy
import json
from utils.util import to_tensor
from dsl.utilities.plot import plot_step
from collections import deque



def overfit_evaluate(
    eval_env,
    agent,
    challenge_key,
    max_episode,
    max_actions,
    warmup,
    max_episode_length,
    state_encoder,
    logger,
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
    agent.epsilon = 1
    agent.depsilon = 1 / (max_actions // 1.5)

    step = 0
    episode = 0
    episode_steps = 0
    episode_reward = 0.0
    episode_positive_rewards = 0
    num_equal_states = 0
    total_actions_taken = 0

    s_t = None

    while episode < max_episode and total_actions_taken < max_actions:
        if s_t is None:
            # Reset to a random training example using reset_overfit
            state, shape = eval_env.reset_overfit(challenge_key, use_test=False)
            s_t = to_tensor(state, device=agent.device, requires_grad=True)
            shape = to_tensor(shape, device=agent.device, requires_grad=True)
            x_t = state_encoder.encode(s_t, shape)

        # Pick action with reduced exploration for fine-tuning
        if step <= warmup:
            action= agent.random_action()
        else:
            # Use lower epsilon for fine-tuning to rely more on pre-trained knowledge
            action= agent.select_action(x_t)

        # Step environment
        (next_state, next_shape), r_t, done, truncated, info = eval_env.step(action)
        next_state = to_tensor(next_state, device=agent.device, requires_grad=True)
        next_shape = to_tensor(next_shape, device=agent.device, requires_grad=True)
        next_x_t = state_encoder.encode(next_state, next_shape)
        total_actions_taken += 1

        # Check for equal states
        if torch.equal(s_t, next_state) and torch.equal(shape, next_shape):
            num_equal_states += 1

        # Count positive reward
        if r_t > 0:
            episode_positive_rewards += 1

        # Check if we hit the max steps in an episode
        if max_episode_length and episode_steps >= max_episode_length - 1 or done:
            truncated = True

        action = torch.tensor(action, device=agent.device, dtype=torch.int64)
        num_actions = torch.tensor(info['num_actions'], device=agent.device, dtype=torch.int64)

        # Observe and update policy
        agent.observe(s_t, shape, x_t, action, r_t, truncated, num_actions)
        if step > warmup:
            agent.update_policy(step)

        step += 1
        episode_steps += 1
        episode_reward += r_t

        # Move to next state
        s_t = next_state
        shape = next_shape
        x_t = next_x_t

        # If the episode ends
        if done or truncated or (total_actions_taken >= max_actions):
            episode_reward = float(round(episode_reward, 2))

            logger.info(
                f"[Overfit Train] Ep:{episode:<4d} | R:{episode_reward:>7.2f} | "
                f"Steps:{episode_steps:>5d} | EqualStates:{num_equal_states:>5d} "
                f"| PosR:{episode_positive_rewards:>5d} | eps:{agent.epsilon:>6.3f}"
            )

            # wandb logging
            if wandb.run is not None:
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
        

    logger.info(f"Training completed on challenge {challenge_key}")

def overfit_evaluate2(
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
            state, shape = eval_env.reset_overfit(challenge_key, use_test=False)
            
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

def test_base_strategy(
    test_env,
    agent,
    challenge_key,
    max_episode_length,
    logger,
    state_encoder,
    negative_streak_threshold=3
):
    """
    Tests overfitting by finding a sequence of transformations that works across all training examples.
    For each training example:
    1. Applies transformations until we get a streak of non-improving cumulative rewards
    2. Keeps the sequence that led to the highest cumulative reward
    3. Resets and applies the successful sequence to the next example
    4. Continues until we find a sequence that works on all examples
    5. Finally applies the successful sequence to the test example
    
    Args:
        test_env: Training environment instance
        agent: WolpertingerAgent
        challenge_key: The specific challenge key to test
        max_episode_length: Maximum steps per episode
        logger: Logger instance
        negative_streak_threshold: Number of consecutive non-improving transformations before stopping
    """
   
    challenge = test_env.challenge_dictionary[challenge_key]
    num_training_examples = len(challenge['train'])
    
    logger.info(f"Starting overfit test on challenge {challenge_key} with {num_training_examples} training examples")
    
    # Initialize variables for tracking successful transformations
    successful_sequence = []
    successful_sequence_rewards = []
    current_sequence = []
    current_sequence_rewards = []
    non_improving_streak = 0
    best_cumulative_reward = float('-inf')
    
    # Test on each training example
    for example_idx in range(num_training_examples):
        logger.info(f"\nTesting on training example {example_idx + 1}/{num_training_examples}")
        
        # Reset environment to this training example
        state, shape = test_env.reset_overfit(challenge_key, use_test=False, example_index=example_idx)
        state = to_tensor(state, device=agent.device, requires_grad=False)
        shape = to_tensor(shape, device=agent.device, requires_grad=False)
        x_t = state_encoder.encode(state, shape)
        
        # If we have a successful sequence from previous examples, apply it first
        if successful_sequence:
            logger.info(f"Applying successful sequence from previous examples: {successful_sequence}")
            for action in successful_sequence:
                (next_state, next_shape), reward, done, truncated, info = test_env.step(action)
                if done:
                    logger.info("Successfully solved with previous sequence!")
                    break
            if done:
                continue
            # Update state after applying successful sequence
            state = to_tensor(next_state, device=agent.device, requires_grad=False)
            shape = to_tensor(next_shape, device=agent.device, requires_grad=False)
            x_t = state_encoder.encode(state, shape)
        
        # Start new sequence search
        current_sequence = []
        current_sequence_rewards = []
        non_improving_streak = 0
        best_cumulative_reward = float('-inf')
        steps = 0
        cumulative_reward = 0
        
        while steps < max_episode_length:
            # Select action
            action = agent.select_action(x_t)
            
            # Apply action
            (next_state, next_shape), reward, done, truncated, info = test_env.step(action)
            next_state = to_tensor(next_state, device=agent.device, requires_grad=False)
            next_shape = to_tensor(next_shape, device=agent.device, requires_grad=False)
            next_x_t = state_encoder.encode(next_state, next_shape)
            
            # Update sequence tracking
            current_sequence = info['actions']
            #current_sequence.append(action)
            current_sequence_rewards.append(reward)
            cumulative_reward += reward
            
            # Check if this transformation improved the cumulative reward
            if cumulative_reward > best_cumulative_reward:
                best_cumulative_reward = cumulative_reward
                non_improving_streak = 0
                # Update successful sequence if this is the best so far
                successful_sequence = current_sequence.copy()
                successful_sequence_rewards = current_sequence_rewards.copy()
            else:
                non_improving_streak += 1
                if non_improving_streak >= negative_streak_threshold:
                    logger.info(f"Stopping due to {negative_streak_threshold} non-improving transformations")
                    # Reset environment to next example and apply successful sequence
                    if example_idx < num_training_examples - 1:
                        logger.info("Moving to next example and applying successful sequence...")
                        break
            
            # Check if solved
            if done:
                logger.info(f"Solved example {example_idx + 1} with sequence: {current_sequence}")
                successful_sequence = current_sequence
                successful_sequence_rewards = current_sequence_rewards
                break
            
            # Move to next state
            state = next_state
            shape = next_shape
            x_t = next_x_t
            steps += 1
        
        # Log results for this example
        logger.info(f"Example {example_idx + 1} results:")
        logger.info(f"Steps taken: {steps}")
        logger.info(f"Best cumulative reward: {best_cumulative_reward:.2f}")
        logger.info(f"Sequence length: {len(current_sequence)}")
        logger.info(f"Successful sequence: {successful_sequence}")
        
        # Log to wandb
        wandb.log({
            "overfit_test/example": example_idx,
            "overfit_test/steps": steps,
            "overfit_test/best_cumulative_reward": best_cumulative_reward,
            "overfit_test/sequence_length": len(current_sequence),
            "overfit_test/solved": done,
            "overfit_test/challenge_key": challenge_key
        })
    
    # After processing all training examples, try the successful sequence on the test example
    logger.info("\nApplying final successful sequence to test example...")
    state, shape = test_env.reset_overfit(challenge_key, use_test=True)
    state = to_tensor(state, device=agent.device, requires_grad=False)
    shape = to_tensor(shape, device=agent.device, requires_grad=False)
    
    test_reward = 0
    for action in successful_sequence:
        (next_state, next_shape), reward, done, truncated, info = test_env.step(action)
        test_reward += reward
        if done:
            logger.info("Successfully solved test example!")
            break
    
    logger.info(f"Test example results:")
    logger.info(f"Final reward: {test_reward:.2f}")
    logger.info(f"Solved: {done}")
    
    # Log test results to wandb
    if wandb.run is not None:
        wandb.log({
            "overfit_test/test_reward": test_reward,
            "overfit_test/test_solved": done,
            "overfit_test/challenge_key": challenge_key
        })
    
    # Log final results
    logger.info(f"\nFinal results for challenge {challenge_key}:")
    logger.info(f"Successful sequence: {successful_sequence}")
    logger.info(f"Sequence rewards: {successful_sequence_rewards}")
    logger.info(f"Sequence length: {len(successful_sequence)}")
    logger.info(f"Test example solved: {done}")
    
    return successful_sequence, successful_sequence_rewards

def test_pruning_strategy(
    test_env,  # ARC environment instance for testing transformations
    agent,     # WolpertingerAgent instance with critics for state evaluation
    challenge_key,  # String identifier for the specific challenge (e.g., "00d62c1b")
    state_encoder,
    max_episode_length,  # Maximum number of steps allowed per episode
    logger,    # Logger instance for tracking progress
    n_depth=3,  # Depth of the tree to build (3^n states will be generated)
    negative_reward_threshold=-1  # Stop a path if cumulative reward goes below this
):
    """
    Tests overfitting by building and pruning a tree of possible transformation sequences.
    For each example:
    1. Start with initial state
    2. Build tree to depth n by:
       - Getting top 3 actions using select_action
       - Applying each action to create new states (branching)
       - Tracking paths and cumulative rewards
    3. After reaching depth n:
       - For each complete path:
         - Get single best action for end state
         - Evaluate Q-value using critic1
         - Combine with cumulative reward
       - Prune to keep top 3 paths
    4. Apply all top 3 paths to next example as starting points
    5. Repeat until all examples processed
    """
    # Get challenge data and count training examples
    challenge = test_env.challenge_dictionary[challenge_key]
    num_training_examples = len(challenge['train'])
    
    logger.info(f"Starting pruning test on challenge {challenge_key} with {num_training_examples} training examples")
    
    # Initialize variables for tracking paths
    current_paths = []  # List of (state, sequence, cumulative_reward) tuples
    best_sequence = None  # Best sequence found so far
    best_reward = float('-inf')  # Best reward achieved
    top_sequences = []  # List to store top 3 sequences from pruning
    
    # Test on each training example
    for example_idx in range(num_training_examples):
        logger.info(f"\nTesting on training example {example_idx + 1}/{num_training_examples}")
        
        # Reset environment to this training example
        state, shape = test_env.reset_overfit(challenge_key, use_test=False, example_index=example_idx)
        state = to_tensor(state, device=agent.device, requires_grad=False)
        shape = to_tensor(shape, device=agent.device, requires_grad=False)
        x_t = state_encoder.encode(state, shape)
        
        # If we have top sequences from previous examples, try each as starting point
        if top_sequences:
            logger.info(f"Trying {len(top_sequences)} sequences from previous examples as starting points")
            starting_points = []
            
            for seq in top_sequences:
                # Reset to initial state for this example
                test_env.state = state
                current_state = state
                current_shape = shape
                current_x_t = x_t
                done = False
                
                # Apply the sequence
                for action in seq:
                    (next_state, next_shape), reward, done, truncated, info = test_env.step(action)
                    if done:
                        logger.info("Successfully solved with previous sequence!")
                        best_sequence = seq
                        best_reward = sum([r for _, r in info['rewards']])
                        break
                    current_state = next_state
                    current_shape = next_shape
                    current_x_t = state_encoder.encode(current_state, current_shape)
                
                if done:
                    break
                
                # Add this as a starting point for tree building
                starting_points.append((current_state, current_shape, current_x_t, seq, sum([r for _, r in info['rewards']])))
            
            if done:
                continue
            
            # Start tree building from each starting point
            current_paths = starting_points
        else:
            # Start with initial state if no previous sequences
            current_paths = [(state, shape, x_t, [], 0.0)]
        
        # Build tree up to depth n
        for depth in range(n_depth):
            logger.info(f"Building tree at depth {depth + 1}/{n_depth}")
            new_paths = []  # List to store expanded paths at current depth
            
            for path_state, path_shape, path_x_t, path_sequence, path_reward in current_paths:
                # Skip paths with negative cumulative reward
                if path_reward < negative_reward_threshold:
                    continue
                
                # Get top 3 moves
                top_actions, top_embedded_actions, q_values = agent.select_top_actions(path_x_t, num_actions=3)
                
                # Apply each action to create new branches
                for action in top_actions:
                    test_env.state = path_state
                    (next_state, next_shape), reward, done, truncated, info = test_env.step(action)
                    next_x_t = state_encoder.encode(next_state, next_shape)
                    if done:
                        logger.info("Successfully solved!")
                        best_sequence = path_sequence + [action]
                        best_reward = path_reward + reward
                        break
                    
                    # Add new path with updated state, sequence, and reward
                    new_paths.append((
                        next_state,
                        next_shape,
                        next_x_t,
                        
                        path_sequence + [action],
                        path_reward + reward
                    ))
                
                if done:
                    break
            
            if done:
                break
            
            current_paths = new_paths
        
        # After building tree to depth n, evaluate and prune paths
        if not done and current_paths:
            scored_paths = []
            for path_state, path_shape, path_x_t, path_sequence, path_reward in current_paths:
                # Get single best action for end state
                best_action, _ = agent.select_action(path_state, path_shape, decay_epsilon=False)
                
                # Get Q-value for this single best action using critic1
                path_state_tensor = to_tensor(path_state, device=agent.device, requires_grad=False)
                with torch.no_grad():
                    next_q = agent.critic1((path_state_tensor, path_shape), to_tensor(best_action, device=agent.device, requires_grad=False)).item()
                
                # Combine cumulative reward with Q-value
                combined_score = path_reward + 0.7 * next_q
                scored_paths.append((path_state, path_shape, path_x_t, path_sequence, path_reward, combined_score))
            
            # Sort paths by combined score and keep top 3
            scored_paths.sort(key=lambda x: x[5], reverse=True)
            current_paths = [(state, shape, x_t, seq, rew) for state, shape, x_t, seq, rew, _ in scored_paths[:3]]
            
            # Store top 3 sequences for next example
            top_sequences = [seq for _, _, _, seq, _, _ in scored_paths[:3]]
            
            # Update best sequence if we found a better one
            if scored_paths and scored_paths[0][5] > best_reward:
                best_sequence = scored_paths[0][3]
                best_reward = scored_paths[0][5]
            
            logger.info(f"Top path scores: {[score for _, _, _, _, _, score in scored_paths[:3]]}")
        
        # Log results for this example
        logger.info(f"Example {example_idx + 1} results:")
        logger.info(f"Number of paths: {len(current_paths)}")
        if current_paths:
            logger.info(f"Best path reward: {current_paths[0][4]:.2f}")
        
        # Log to wandb
        wandb.log({
            "pruning_test/example": example_idx,
            "pruning_test/num_paths": len(current_paths),
            "pruning_test/best_reward": current_paths[0][4] if current_paths else float('-inf'),
            "pruning_test/solved": done,
            "pruning_test/challenge_key": challenge_key
        })
    
    # After processing all training examples, try the best sequence on the test example
    logger.info("\nApplying best sequence to test example...")
    state, shape = test_env.reset_overfit(challenge_key, use_test=True)
    state = to_tensor(state, device=agent.device, requires_grad=False)
    shape = to_tensor(shape, device=agent.device, requires_grad=False)
    agent.reset(state, shape)
    
    test_reward = 0
    for action in best_sequence:
        (next_state, next_shape), reward, done, truncated, info = test_env.step(action)
        test_reward += reward
        if done:
            logger.info("Successfully solved test example!")
            break
    
    logger.info(f"Test example results:")
    logger.info(f"Final reward: {test_reward:.2f}")
    logger.info(f"Solved: {done}")
    
    # Log test results to wandb
    wandb.log({
        "pruning_test/test_reward": test_reward,
        "pruning_test/test_solved": done,
        "pruning_test/challenge_key": challenge_key
    })
    
    # Log final results
    logger.info(f"\nFinal results for challenge {challenge_key}:")
    logger.info(f"Best sequence: {best_sequence}")
    logger.info(f"Sequence length: {len(best_sequence)}")
    logger.info(f"Test example solved: {done}")
    
    return best_sequence, test_reward





