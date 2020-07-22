import torch
import numpy as np
from numpy.random import default_rng
rng = default_rng()
STATE_SPACE_SIZE = 4
ACTION_SPACE_SIZE = 5
ACTION_SAMPLE_SIZE = 4

class Environment(object):
    def __init__(
        self,
        state_space = STATE_SPACE_SIZE,
        action_space_size = ACTION_SPACE_SIZE,
        action_sample_size = ACTION_SAMPLE_SIZE
    ):
        self.state_space = state_space
        self.action_space_size = action_space_size
        self.action_sample_size = action_sample_size
        self.behavior_matrix = self._get_behavior_matrix()

    def _get_behavior_matrix(self):
        behavior_matrix = torch.tensor(np.array([[.7, 0, .2, .1, 0],
                        [.3, .7, 0, 0, 0],
                        [.2, 0, .7, 0, .1],
                        [0, .1, .2, 0, .7]]), dtype=torch.float32)
        return behavior_matrix

    def _sample_utterance(self):
        """
        :description: Randomly generates an intent confidence vector
        :return: nx1 vector that sums up to 1
        """
        values = torch.zeros(self.state_space, dtype=torch.float32)
        unnormalized_state = values.log_normal_()
        normalized = values/values.sum()
        utterance_vector = normalized.reshape(1, normalized.shape[0])
        return utterance_vector

    def _sample_actions(self):
        """
        :description: Randomly samples potential actions to take
        :return: set of potential actions to take
        """
        sampled_actions = rng.integers(self.action_space_size, size=self.action_sample_size)
        action_space = set(sampled_actions)
        return action_space

    def _sample_reward(self, action): # @todo see how to format this as tensor only
        """
        :description: Returns reward value probabilistically based on utterance vector and action taken
        """
        UR_matrix = self.state.matmul(self.behavior_matrix)
        UR = UR_matrix[0][action.item()]
        reward = torch.zeros(UR.shape, dtype=torch.float32)
        reward = reward.bernoulli_(UR)
        return reward, UR_matrix

    def reset(self):
        self.state = self._sample_utterance()
        self.actions = self._sample_actions()
        return self.state, self.actions

    def step(self, action):
        reward,_ = self._sample_reward(action)
        self.state = self._sample_utterance()
        return self.state, reward, True, 'Chat environment' #From openAI gym: observation, reward, done, info
    
    def get_action_space_size(self):
        return self.action_space_size

    def get_state_space_size(self):
        return self.state_space