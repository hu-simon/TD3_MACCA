"""
Written by Simon Hu, all rights reserved.
"""

import torch
import random
import numpy as np
from torch.autograd import Variable

class ReplayBuffer():
    def __init__(self, env, init_length=1000, capacity=1e6):
        """
        Initializes the replay buffer.

        Parameters
        ----------
        env: gym environment object
            Object representing the environment.
        init_length : int, optional
            Number of transitions to collect at initialization.
        capacity : int, optional
            Maximum size of the replay buffer before the index resets.
        """
        self.capacity = capacity
        self.buffer = list()
        self.index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Figure out how to collect samples for initialization.

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

    def sample_from_buffer(self, batch_size=64):
        """
        Samples points ffrom the buffer.

        Parameters
        ----------
        batch_size : int, optional
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
            Tensor representing the next state sampled from the buffer.
        done_tensor : tensor
            Tensor representing the done information sampled from the buffer.
        """
        sampled_batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*sampled_batch))

        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        reward_tensor = torch.unsqueeze(torch.FloatTensor(reward),1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        done_tensor = torch.unsqueeze(torch.FloatTensor(done),1).to(self.device)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
