"""
Written by Simon Hu, all rights reserved.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, env, hidden_size=[256], discount_factor=0.99, learning_rate=1e-4):
        """
        Initializes the network for the Actor.

        Parameters
        ----------
        env : gym environment object
            Object representing the gym environment.
        hidden_size : list of ints, optional
            List containing the sizes of the hidden layers.
        discount_factor : double, optional
            Discount factor used in computing the discounted rewards.
        learning_rate : double, optional
            Learning rate of the network.
        """
        super(Actor, self).__init__()

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(self.state_dim, self.hidden_size[0])
        # Figure out how to set up the hidden layers.
        self.output_layer = nn.Linear(self.hidden_size[-1], self.action_dim)
        # Figure out how to set up the batchnorm layers.
        self.batchnorm_output = nn.BatchNorm1d(self.action_dim)
        self.tanh_layer = nn.Tanh()

        self.loss = MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        """
        Performs the forward pass for the network, given the state.

        Parameters
        ----------
        state : tensor
            Tensor representing the state of the environment.
        action : tensor
            Tensor representing the action to be taken.

        Returns
        -------
        output : tensor
            Tensor representing the output of the network, i.e. the action to take.
        """
        # TODO Need to put batch layers everywhere; the output is already done.
        x = F.relu(self.input_layer(state))
        # Figure out how to incorporate the hidden layers.
        output = self.tanh_layer(self.batchnorm_output(self.output_layer(x)))
        return output

def Critic(nn.Module):
    def __init__(self, env, hidden_size=[256], discount_factor, learning_rate=1e-3):
        """
        Initializes the network for the Critic.

        Parameters
        ----------
        env : gym environment object
            Object representing the gym environment.
        hidden_size : list of ints, optional
            List containing the sizes of hidden layers.
        discount_factor : double, optional
            Discount factor used in computing the discounted rewards.
        learning_rate : double, optional
            Learning rate of the network.
        """
        super(Critic, self).__init__()

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        # Network for the first critic.
        self.input_layer1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_size[0])
        # Figure out how to set up the hidden layers.
        self.output_layer1 = nn.Linear(self.hidden_size[-1], 1)
        # Figure out how to set up a batch norm for the first hidden layer.

        # Network for the second critic.
        self.input_layer2 = nn.Linear(self.state_dim + self.action_dim, self.hidden_size[0])
        # Figure out how to set up the hidden layers.
        self.output_layer2 = nn.Linear(self.hidden_size[-1], 1)
        # Figure out how to set up a batch norm for the first hidden layer.

    def forward(self. state, action):
        """
        Performs the forward pass for the network, given a state and action.

        Parameters
        ----------
        state : tensor
            Tensor representing the state of the environment.
        action : tensor
            Tensor representing the action to be taken.

        Returns
        -------
        output : tensor
            Tensor representing the output of the network.
        """
        # TODO incorporate the batch norm layer into the forward pass.
        x = torch.cat([state, action], 1)
        x = F.relu(self.input_layer1(x))
        # Figure out how to incorporate the hidden layers.
        output = F.relu(self.output_layer1(x))
        return output

    def forward_both(self, state, action):
        """
        Performs the forward pass for the network, given a state and action.
        Note that this function returns the output for both networks, since
        we want to reduce the evaluation time.

        Parameters
        ----------
        state : tensor
            Tensor representing the state of the environment.
        action : tensor
            Tensor representing the action to be taken.

        Returns
        -------
        output : tensor
            Tensor representing the output of the network.
        """
        # TODO incorporate the batch norm layer into the forward pass.
        x = torch.cat([state, action], 1)

        x1 = F.relu(self.input_layer1(x))
        # Figure out how to incorporate the hidden layers.
        output1 = self.output_layer1(x1)

        x2 = F.relu(self.input_layer2(x))
        # Figure out how to incorporate the hidden layers.
        output2 = self.output_layer2(x2)

        return output1, output2
