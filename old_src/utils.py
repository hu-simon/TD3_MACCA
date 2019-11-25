"""
Written by Simon Hu, all rights reserved.
"""

import gym
import time
import torch
import numpy as np

def evaluate_policy(env_name, policy, seed=2019, num_episodes=10):
    """
    Evaluates the trained policy.

    Parameters
    ----------
    env_name : string
        String representing the environment to be evaluated.
    policy : DDPG object
        DDPG object containing the network to perform the evaluation with.
    seed : int, optional
        Random seed.
    num_episodes : int, optional
        The number of episodes to perform evaluation.
    """
    evaluation_env = gym.make(env_name, rand_init=False)
    evaluation_env.seed(seed)

    total_reward = 0

    state = evaluation_env.reset()
    done = False
    for _ in range(num_episodes):
        action = policy.actor.forward(torch.FloatTensor(state)).detach().numpy()
        state, reward, done, _ = evaluation_env.step(action)
        total_reward += reward
        if done:
            state = evaluation_env.reset()
            break

    avg_reward = total_reward
    print("Evaluation over {} episodes: {}".format(num_episodes, avg_reward))
    #evaluation_env.close()
    return avg_reward
