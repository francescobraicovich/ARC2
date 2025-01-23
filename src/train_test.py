from utils.util import to_tensor
import numpy as np
import torch
from dsl.utilities.plot import plot_step

def train(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, save_per_epochs):
    agent.is_training = True
    step = episode = episode_steps = episode_positive_rewards = num_equal_rewards = 0
    episode_reward = 0.
    s_t = None
    while episode < max_episode:
        while True:
            if s_t is None:
                s_t, shape = env.reset()
                s_t = to_tensor(s_t, device=agent.device)
                shape = to_tensor(shape, device=agent.device)
                agent.reset(s_t, shape)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if step <= warmup:
                action, embedded_action = agent.random_action()
            else:
                action, embedded_action = agent.select_action(s_t, shape)
            #actions[episode_steps] = action

            # env response with next_observation, reward, terminate_info
            state1, r_t, done, truncated, info = env.step(action)
            s_t1, shape1 = state1
            s_t1, shape1 = to_tensor(s_t1, device=agent.device), to_tensor(shape1, device=agent.device)
            
            if torch.equal(shape, shape1):
                if torch.equal(s_t, s_t1):
                    num_equal_rewards += 1
        
            if r_t > 0:
                episode_positive_rewards += 1

            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True

            # agent observe and update policy
            agent.observe(r_t, s_t1, shape1, done)
            if step > warmup:
                agent.update_policy()

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1
            shape = shape1

            if done or truncated:  # end of an episode
                episode_reward = round(episode_reward, 2)
                logger.info(
                    "Ep:{:<4} | R:{:>7.2f} | Steps:{:>5} | Equal:{:>5} | Rs>0:{:>5} | eps:{:>6.3f}".format(
                        episode, episode_reward, episode_steps, num_equal_rewards, episode_positive_rewards, agent.epsilon
                    )
                )

                agent.memory.append(
                    s_t,
                    shape,
                    agent.select_action(s_t, shape)[1], # embedded action only
                    0., True
                )

                # reset
                s_t = None
                episode_steps =  0
                num_equal_rewards = 0
                episode_positive_rewards = 0
                episode_reward = 0.
                episode += 1
                #actions = np.zeros((max_episode_length, 3))
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % save_per_epochs == 0:
            agent.save_model(save_model_dir)
            logger.info(f"### Model Saved in {save_model_dir} before Ep:{episode} ###")

def test(env, agent, model_path, test_episode, max_episode_length, logger):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    episode_steps = 0
    episode_reward = 0.
    s_t = None
    for i in range(test_episode):
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)

            action = policy(s_t)
            s_t, r_t, done, _ = env.step(action)
            episode_steps += 1
            episode_reward += r_t
            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True
            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward)
                )
                s_t = None
                break