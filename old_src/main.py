"""
Written by Simon Hu, all rights reserved.
"""

import numpy as np
import torch
import gym
import utils
from td3 import TD3

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set up the environment.
    env_name = "modified_gym_env:ReacherPyBulletEnv-v1"
    env = gym.make(env_name, rand_init=False)

    #env.render(mode="human")
    #time.sleep(3)
    #env = gym.make(env_name)

    # Set the seeds.
    seed = 2019
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment parameters.
    state_dim = env.observation_space.shape[0] - 1
    action_dim = env.action_space.shape[0]

    # Training hyperparameters.
    num_steps = 2e5
    discount_factor = 0.99
    init_length = 1000
    batch_size = 256
    eval_index = 1000

    policy = TD3(env)

    policy_evaluations = list()

    state = env.reset()
    done = False
    for t in range(int(num_steps)):

        action = policy.convert_action(np.array(state)) + np.random.normal(0, 0.1, size=action_dim)
        next_state, reward, done, _ = env.step(action)

        # Store the experience into the buffer.
        policy.buffer.add_to_buffer([state, action, reward, next_state, done])

        state = next_state
        episode_reward += reward

        # Perform a policy update when we have collected sufficient data.
        policy.train()
        if (t+1) % eval_index == 0:
            print(t+1)
            policy_evaluations.append(utils.evaluate_policy(env_name, policy, seed=seed, num_episodes=30))

        if done:
            # Reset everything.
            state = env.reset()
            done = False
