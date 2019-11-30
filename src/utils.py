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
    policy : TD3 object
        TD3 object containing the network to perform the evaluation with.
    seed : int, optional
        Random seed for the environment.
    num_episodes : int, optional
        Number of episodes to perform evaluation.
    """
    pass
