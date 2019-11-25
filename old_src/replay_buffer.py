"""
Written by Simon Hu, all rights reserved.
"""

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class ReplayBuffer():
    def __init__(self, env, init_length=1000, capacity=10000):
        """
        Initializes the replay buffer.

        Parameters
        ----------
        env : gym environment object
            Object representing the environment.
        init_length : int, optional
            Number of transitions to collect at initialization.
        capacity : int, optional
            Maximum size of the replay buffer before the index resets.
        """
        self.capacity = capacity
        self.buffer = list()
        self.index = 0

        # Collect samples for the initialization.
        state = env.reset()
        for _ in range(init_length):
            action = env.action_space.sample() + np.random.normal(0, 0.1, size=env.action_space.shape[0])
            next_state, reward, done, _ = env.step(action)
            transition = [state, action, reward, next_state, done]
            self.add_to_buffer(transition)
            if done:
                state = env.reset()
            else:
                state = next_state

    def add_to_buffer(self, transition):
        """
        Adds a transition object into the buffer.

        Parameters
        ----------
        transition : list
            List containing the state, action, reward, next_state, and done flag.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample_from_buffer(self, batch_size):
        """
        Samples points from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw from the buffer.

        Returns
        -------
        state_tensor : tensor
            Tensor representing the current states sampled from the buffer.
        action_tensor : tensor
            Tensor representing the current actions sampled from the buffer.
        reward_tensor : tensor
            Tensor representing the current rewards sampled from the buffer.
        next_state_tensor : tensor
            Tensor representing thet next state sampled from the buffer.
        done_tensor : tensor
            Tensor representing the done information sampled from the buffer.
        """
        sampled_batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*sampled_batch))

        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        reward_tensor = torch.unsqueeze(torch.FloatTensor(reward),1)
        next_state_tensor = torch.FloatTensor(next_state)
        done_tensor = torch.unsqueeze(torch.FloatTensor(done),1)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
