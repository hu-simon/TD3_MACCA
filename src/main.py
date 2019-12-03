"""
Written by Simon Hu, all rights reserved.
"""

# Figure out which import statements you need.

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("GPU:", use_cuda)

    # Set up the environment here.

    # Set the seeds.
    seed = 2019

    # Environment parameters.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Training hyperparameters.
    num_steps = 2e5
    discount_factor = .99
    init_length = 1000
    batch_size = 256
    eval_index = 1000

    
