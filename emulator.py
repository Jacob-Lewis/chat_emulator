from numpy.random import default_rng
import numpy as np
import torch
rng = default_rng()

def get_reward_matrix():
    reward = np.array([[.4, .3, .2, .1, 0],
                    [.3, .4, .2, .1, 0],
                    [.2, .3, .4, 0, .1],
                    [0, .1, .2, .3, .4]], dtype=float)
    reward_matrix = torch.tensor(reward, dtype=torch.float32)
    return reward_matrix

def sample_utterance(state_space):
    """
    :description: Randomly generates an intent confidence vector
    :return: nx1 vector that sums up to 1
    """
    vals = rng.random(state_space)
    vals = vals.astype(np.float)
    normalized = vals/sum(vals)
    max_arg = normalized.argmax()
    min_arg = normalized.argmin()
    normalized[max_arg] += normalized[min_arg]
    normalized[min_arg] -= normalized[min_arg]
    utterance_vector = normalized.reshape(1, normalized.shape[0])
    torchified = torch.tensor(utterance_vector, dtype=torch.float32)
    return torchified


def sample_actions(range_size, sample_size):
    """
    :description: Randomly samples potential actions to take
    :return: set of potential actions to take
    """
    sampled_actions = rng.integers(range_size, size=sample_size)
    action_space = set(sampled_actions)
    return action_space

def sample_reward(utterance_vector, action):
    """
    :description: Returns reward value probabilistically based on utterance vector and action taken
    """
    reward_matrix = get_reward_matrix()
    UR_matrix = utterance_vector.matmul(reward_matrix)
    try:
        action = action.type(torch.LongTensor)
    except: 
        pass
    UR = UR_matrix[0][action]
    reward = torch.tensor(np.random.binomial(1, UR), dtype=torch.float32)
    reward = abs(reward - 1) #Converts to loss function
    return reward, UR_matrix