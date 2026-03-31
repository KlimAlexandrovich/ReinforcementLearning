import torch
import torch.nn as nn
import numpy as np


class DQNAgent(nn.Module):
    """
    Provides an implementation of a deep Q-learning agent using a Q-network.
    This class defines a reinforcement learning agent that estimates Q-values using a neural network.
    It supports action sampling using an epsilon-greedy policy and includes functionality for both
    training and inference. The agent's policy incorporates exploration to balance the tradeoff
    between exploiting known information and exploring new possibilities.
    Methods:
        forward: Evaluates the Q-values for the given states using the Q-network.
        get_q_values: Computes Q-values for the given states using the current model.
        sample_actions_by_q_values: Selects actions based on Q-values using an epsilon-greedy policy.
        sample_actions: Samples actions based on the provided states and the greediness setting.
    """

    def __init__(self, q_network: nn.Module, eps: float = 0.5) -> None:
        super().__init__()
        self.eps = eps
        self.q_network = q_network

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """ Evaluates the Q-values for the given states using the Q-network. """
        q_values = self.q_network(states)
        return q_values

    @torch.no_grad()
    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """
        Computes Q-values for the given states using the current model.
        The computations are performed without tracking gradients
        to ensure efficient evaluation during testing or inference on numpy arrays.
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(np.array(states), device=model_device, dtype=torch.float32)
        q_values = self.forward(states=states).data.cpu().numpy()
        return q_values

    def sample_actions_by_q_values(self, q_values: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Selects actions based on Q-values using an epsilon-greedy policy. This method either returns
        the greedy actions (i.e., actions with the highest Q-values) or incorporates exploration
        by choosing random actions with a probability defined by the epsilon parameter.

        Note: When using Noisy Networks, set eps=0.0 and the network handles exploration automatically.
        """
        greedy_actions = q_values.argmax(axis=-1)
        if greedy or self.eps == 0.0: return greedy_actions
        batch_size, n_actions = q_values.shape
        random_actions = np.random.randint(0, n_actions, size=batch_size)
        should_explore = np.random.binomial(n=1, p=self.eps, size=batch_size)
        epsilon_greedy_actions = np.where(should_explore, random_actions, greedy_actions)
        return epsilon_greedy_actions

    def sample_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Samples actions based on the provided states and the greediness setting. The method calculates
        Q-values for the given states and uses those Q-values to determine the resulting actions. This
        function supports both greedy and non-greedy action sampling depending on the specified input.
        """
        # TODO: Add asserts for states.
        q_values = self.get_q_values(states)
        actions = self.sample_actions_by_q_values(q_values=q_values, greedy=greedy)
        return actions
