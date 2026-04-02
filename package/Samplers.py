import numpy as np
import gymnasium as gym
from .DQNAgent import DQNAgent


class ActionSampler:
    """ A protocol which defines a "Callable" which samples actions from states. """

    def __call__(self, state: np.ndarray) -> int:
        raise NotImplemented


class RandomActionSampler(ActionSampler):
    """
    A sampler class for generating random actions within a given action space.
    This class is designed to work with environments following the "OpenAI Gym interface".
    It takes an action space as input during initialization and provides a callable interface
    to sample random actions based on the provided action space. The primary use case is for
    testing or exploratory purposes, where random actions need to be generated regardless
    of the given state.
    """

    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space

    def __call__(self, state: np.ndarray) -> int:
        assert isinstance(state, np.ndarray), f"State is not a numpy array, got {type(state)}."
        assert state.ndim == 3, f"Not compatible with state of shape {state.shape}, expected 3 dimensions."
        action = self.action_space.sample()
        return action


class DQNActionSampler(ActionSampler):
    """
    DQNAgent works on batched np.ndarray inputs.
    This class uses a "DQNAgent" to sample actions from single LazyFrames observations.
    """

    def __init__(self, agent: DQNAgent, greedy: bool = True):
        self.agent = agent
        self.greedy = greedy

    def __call__(self, state: np.ndarray) -> int:
        assert isinstance(state, np.ndarray), f"State is not a numpy array, got {type(state)}."
        assert state.ndim == 3, f"Not compatible with state of shape {state.shape}, expected 3 dimensions."
        state_batched = np.expand_dims(np.array(state), axis=0)
        action_batched = self.agent.sample_actions(states=state_batched, greedy=self.greedy)
        action = action_batched.item()
        return action
