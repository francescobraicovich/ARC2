import os
import numpy as np
import torch
import wandb
import copy

from utils.util import to_tensor

from dsl.utilities.plot import plot_step

state_encoder = lambda state, shape: torch.ravel(state).float()[:256]

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
    shape = None
    x_t = None

    positive_rewards_history = []
    rewards_history = []

    while episode < max_episode and total_actions_taken < max_actions:
        if s_t is None:
            state, shape = train_env.reset()
            s_t = to_tensor(state, device=agent.device, requires_grad=True)
            shape = to_tensor(shape, device=agent.device, requires_grad=True)
            x_t = state_encoder(s_t, shape)
            agent.reset(x_t)

        # Pick action
        if step <= warmup:
            action = agent.random_action()
            assert type(action) == int, "Action should be an integer but got: {}".format(type(action))
        else:
            action = agent.select_action(x_t)
            assert type(action) == int, "Action should be an integer but got: {}".format(type(action))
        

        # Step environment
        (next_state, next_shape), r_t, done, truncated, info = train_env.step(action)
        next_state = to_tensor(next_state, device=agent.device, requires_grad=True)
        next_shape = to_tensor(next_shape, device=agent.device, requires_grad=True)
        next_x_t = state_encoder(next_state, next_shape)

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
        action = torch.tensor(action, device=agent.device, dtype=torch.int64)
        agent.observe(s_t, shape, x_t, action, r_t, done)
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
            s_t, shape = None, None
            x_t = None
            episode_steps = 0
            episode_reward = 0.0
            episode_positive_rewards = 0
            num_equal_states = 0
            episode += 1

            # NOTE: Siamo rimasti qui

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
    saved_epsilon = agent.epsilon # Save epsilon value for restoration after evaluation
    agent.epsilon = 0.0 # Set epsilon to 0 for evaluation so that no exploration is performed
    agent.eval()

    total_rewards = []

    for ep in range(episodes):
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
        logger.info(f"[Eval] Ep:{ep+1}/{episodes} | Reward: {episode_reward:.2f} | Steps: {steps} | PosR: {episode_positive_rewards} | EqualStates: {num_equal_states}")

    avg_eval_reward = np.mean(total_rewards)
    logger.info(f"[Eval] Average Reward over {episodes} episodes: {avg_eval_reward:.2f}")

    # wandb logging
    wandb.log({
        "eval/episodes": episodes,
        "eval/average_reward": avg_eval_reward
    })

    # Switch agent back to training mode
    agent.is_training = True
    agent.epsilon = saved_epsilon # Restore epsilon value

def Test_1(agent, env, max_episode_length, logger, max_iterations=1000):
    """
    Custom test function that trains on a single ARC problem until complete overfitting.
    Each episode begins with reloading the base pretrained weights.
    The function logs the challenge key and the sequence of actions applied.
    """
    # Save base pretrained weights
    base_actor_state = copy.deepcopy(agent.actor.state_dict())
    base_critic1_state = copy.deepcopy(agent.critic1.state_dict())
    base_critic2_state = copy.deepcopy(agent.critic2.state_dict())
    
    # Obtain fixed ARC problem
    initial_state, initial_shape = env.reset()
    fixed_challenge_key = env.info['key']
    logger.info(f"Fixed challenge key: {fixed_challenge_key}")

    solved = False
    episode = 0
    optimal_actions = None

    while not solved and episode < max_iterations:
        # Reset agent to base pretrained weights for each episode
        agent.actor.load_state_dict(base_actor_state)
        agent.critic1.load_state_dict(base_critic1_state)
        agent.critic2.load_state_dict(base_critic2_state)
        
        # Reset environment to fixed problem
        current_state, current_shape = initial_state, initial_shape
        agent.reset(
            to_tensor(current_state, device=agent.device, requires_grad=True),
            to_tensor(current_shape, device=agent.device, requires_grad=True)
        )
        actions_sequence = []
        done = False
        steps = 0

        while not (done or steps >= max_episode_length):
            # Select action without decaying exploration
            action, _ = agent.select_action(
                to_tensor(current_state, device=agent.device, requires_grad=True),
                to_tensor(current_shape, device=agent.device, requires_grad=True),
                decay_epsilon=False
            )
            actions_sequence.append(action)
            (next_state, next_shape), reward, done, truncated, info = env.step(action)
            current_state, current_shape = next_state, next_shape
            steps += 1
            if done:
                solved = True
                optimal_actions = actions_sequence
                logger.info(f"Solved in episode {episode} after {steps} steps.")
                logger.info(f"Optimal sequence of actions: {actions_sequence}")
                logger.info(f"Challenge key: {fixed_challenge_key}")
                break

        if not solved:
            logger.info(f"Episode {episode} ended without solution. Restarting from base model.")
        episode += 1

    return fixed_challenge_key, optimal_actions


def Test_2(agent, env_list, max_episode_length, logger, max_iterations=1000):
    """
    Inferencial test function that evaluates the trained model on multiple instances of the same challenge key.
    
    Requisites:
        - Inference only: the model predicts the next action without performing any training.
        - A partially found sequence of actions (global_sequence) is used, which if it produces positive rewards on an example,
          is applied to other examples to verify its generalizability.
        - If the partial sequence fails (receives non-positive reward) on an example, the extension on that environment is interrupted
          and another instance is tried.
        - The cycle continues until a complete sequence that solves the environment is found or until the maximum number of iterations is reached. 
    
    Args:
      agent: Trained agent.
      env_list: List of environments (instances) sharing the same challenge key.
      max_episode_length: Maximum number of steps for extending the sequence in each attempt.
      logger: Logger instance.
      max_iterations: Maximum total number of iterations to try.
      
    Returns:
      A tuple: containing the challenge key and the global sequence of actions found (if complete).
    """
    # Set the agent to evaluation mode
    agent.eval()
    challenge_key = env_list[0].info['key']
    logger.info(f"[Test_2] Challenge key: {challenge_key}")
    
    # Global sequence that will be extended progressively
    global_sequence = []
    iteration = 0
    
    # Main loop, which terminates when max_iterations is reached
    while iteration < max_iterations:
        # Switch between instances in round-robin fashion
        for env in env_list:
            iteration += 1
            logger.info(f"[Test_2] Itaration {iteration} on new episode.")
            
            # Reset of the environment and obtain the initial state
            state, shape = env.reset()
            
            # Replay of the already found global sequence to bring the environment to the corresponding state
            valid_sequence = True
            for action in global_sequence:
                (state, shape), reward, done, truncated, info = env.step(action)
                # If a transformation does not produce positive reward, the sequence is not valid on this example
                if reward <= 0:
                    logger.info("[Test_2] The partial sequence did not produce a positve reward; skipping this example.")
                    valid_sequence = False
                    break
                # If the environment is already solved during the replay, we return the found sequence
                if done:
                    logger.info("[Test_2] Episode solved using the exiusting sequence!")
                    return challenge_key, global_sequence
            
            # If the global sequence does not work on this example, we move on to the next one
            if not valid_sequence:
                continue
            
            # Preparing the current state to extend the sequence
            state_tensor = to_tensor(state, device=agent.device, requires_grad=False)
            shape_tensor = to_tensor(shape, device=agent.device, requires_grad=False)
            agent.reset(state_tensor, shape_tensor)
            
            steps = 0
            # Try to extend the sequence starting from the reached state
            while steps < max_episode_length:
                # Prediction of the action in inference mode (without epsilon decay)
                action, _ = agent.select_action(state_tensor, shape_tensor, decay_epsilon=False)
                (next_state, next_shape), reward, done, truncated, info = env.step(action)
                logger.info(f"[Test_2] Step {steps} - reward: {reward}")
                
                # If the action is not valid, we interrupt the extension on this example
                if reward > 0:
                    global_sequence.append(action)
                    logger.info(f"[Test_2] Extended sequence: {global_sequence}")
                else:
                    logger.info("[Test_2] Invalid action (negative reward). Interrupting extension on this example.")
                    break
                
                # If the environment signals completion, we have found a solution
                if done:
                    logger.info("[Test_2] Episode solved with the exteded sequence!")
                    return challenge_key, global_sequence
                
                # Update the state for the next step
                state_tensor = to_tensor(next_state, device=agent.device, requires_grad=False)
                shape_tensor = to_tensor(next_shape, device=agent.device, requires_grad=False)
                steps += 1

    logger.info("[Test_2] No complete solution found, maximum iteration number reached.")
    return challenge_key, global_sequence