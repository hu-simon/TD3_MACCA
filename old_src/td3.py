"""
Written by Simon Hu, all rights reserved.
"""

import copy
import numpy as np
import torch.optim as optim
from actor_critic_td3 import Actor, Critic
from replay_buffer import ReplayBuffer
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3():
    def __init__(self, env, init_length=1000, actor_lr=1e-4, critic_lr=1e-3, discount_factor=0.99, batch_size=256, tau=0.005,
        policy_delay=2, policy_noise=0.2, max_noise=0.5):
        """
        Initializes the TD3 object.

        Parameters
        ----------
        env : gym environment object
            Object representing the gym environment.
        init_length : int, optional
            Number of transitions to collect at initialization.
        actor_lr : double, optional
            Learning rate of the actor network.
        critic_lr : double, optional
            Learning rate of the critic network.
        discount_factor : double, optional
            Discount factor used for computing the discounted rewards.
        batch_size : int, optional
            Batch size used during training.
        tau : double, optional
            Parameter determining the size of the soft update.
        policy_delay : int, optional
            The frequency of the delayed policy updates.
        policy_noise : double, optional
            Amount of noise to be added to the target policy during training.
        max_noise : double, optional
            The maxmimum amount of noise allowed for the target policy.
        """
        self.state_dim = env.observation_space.shape[0]-1
        self.action_dim = env.action_space.shape[0]
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.max_noise = max_noise

        self.actor = Actor(env).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(env).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.paremeters(), lr=3e-4)

        self.total_iterations = 0

        self.buffer = ReplayBuffer(env, init_length=init_length)

    def convert_action(self, action):
        """
        Converts a state into an action.

        Parameters
        ----------
        action : tensor
            Tensor representing the action to take.
        """
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action = self.actor.forward(state).cpu().data.numpy().flatten()
        return action

    def soft_update(self, target_model, source_model, tau=0.005):
        """
        Performs a soft update of the weights of the target network, using the source network.

        Parameters
        ----------
        target_model : neural network
            The target network whose weights are to be updated.
        source_model : neural network
            The source network used to update the target network.
        tau : double, optional
            Weighting parameter used to perform the update.
        """
        for target_parameter, source_parameter in zip(target_model.parameters(), source_model.parameters()):
            target_parameter.data.copy_(tau * source_parameter.data + (1 - tau) * target_parameter.data)

    def update_target_networks(self):
        """
        Updates the target networks.
        """
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def train(self):
        """
        Trains the policy.
        """
        self.total_iterations += 1

        # Sample from the replay buffer.
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.buffer.sample_from_buffer(self.batch_size)
        with torch.no_grad():
            # Need to pick the action based on a noisy version of the policy, clipped by the maximum allowable noise.
            noise = torch.randn_like(batch_action) * self.policy_noise
            noise = noise.clamp(-self.max_noise, self.max_noise)
            action = self.actor_target.forward(batch_next_state) + noise

            Q1_target, Q2_target = self.critic_target.forward_both(batch_next_state, batch_action)
            Q_target = torch.min(Q1_target, Q2_target) # take the minimum.
            Q_prime = reward + ((1. - batch_done) * self.discount_factor * Q_target)

        Q1_estimate, Q2_estimate = self.critic.forward_both(batch_state, batch_action)

        # Compute the critic loss and take an optimization step.
        loss_critic = F.mse_loss(Q1_target, Q1_estiamte) + F.mse_loss(Q2_target, Q2_estimate)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Compute the delayed policy updates.
        if self.total_iterations % self.policy_delay == 0:
            # Compute the actor loss and perform an optimization step.
            loss_actor = -self.critic.forward(batch_state, self.actor(batch_state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            # Perform a soft update of the network parameters.
