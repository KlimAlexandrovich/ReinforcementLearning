import torch
from torch import nn
from typing import Callable, Literal, Optional


class TDLoss(nn.Module):
    """ Defines a temporal difference loss computation class for reinforcement learning.
        This class implements the temporal difference (TD) loss required for training reinforcement
        learning agents. It supports both standard and double DQN methods for calculating the TD error.
        The loss function helps to minimize the discrepancy between the predicted Q-values from the
        current agent and the target Q-values based on the next state. """

    def __init__(
            self,
            agent: nn.Module,
            target_agent: nn.Module,
            gamma: float = 0.99,
            method: Literal["vanilla", "double"] = "vanilla"
    ):
        super().__init__()
        self.agent = agent
        self.target_agent = target_agent
        self.gamma = gamma
        self.method: Literal["vanilla", "double"] = method
        self._fn: Callable = self._get_fn(method=self.method)

    def _get_fn(self, method: str) -> Callable:
        """ Retrieves the appropriate loss function based on the provided method name. """
        return dict(vanilla=self._regular_dqn_loss, double=self._double_dqn_loss)[method]

    def _regular_dqn_loss(
            self, s: torch.Tensor, a: torch.Tensor, ns: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Computes the loss for a regular Deep Q-Network (DQN) by comparing the predicted
            Q-values for the current state-action pairs to the target Q-values derived
            from the next states. """
        prediction = self.agent(s)  # Q(s, a), shape: [B, A]
        prediction = prediction.gather(dim=1, index=a.unsqueeze(-1)).squeeze(-1)  # Filtered by actions.
        with torch.no_grad():
            target = self.target_agent(ns)  # Q(s`, a`)
            target = target.max(dim=1).values  # arg max a (Q(s`, a`))
        assert prediction.shape == target.shape, f"{target.shape} != {prediction.shape}"
        return prediction, target

    def _double_dqn_loss(self, s: torch.Tensor, a: torch.Tensor, ns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Calculate the Double DQN loss by leveraging both the online and target networks.
            This function computes the prediction of the online network for the current states
            filtered by the selected actions and calculates the target values using the target
            network and the online network's predicted next actions. The resulting outputs are returned
            to be used in a loss computation. """
        prediction = self.agent(s)  # Q(s, a), shape: [B, A]
        prediction = prediction.gather(dim=1, index=a.unsqueeze(-1)).squeeze(-1)  # Filtered by actions, shape: [B]
        with torch.no_grad():
            # 1) Action selection by ONLINE net: a* = arg max a Q_online(s', a).
            nqv_online = self.agent(ns)  # [B, A]
            next_actions = nqv_online.argmax(dim=1)  # shape: [B]
            # 2) Action evaluation by TARGET net: Q_target(s', a*).
            nqv_target = self.target_agent(ns)  # shape: [B, A]
            target = nqv_target.gather(dim=1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # shape: [B]
        assert prediction.shape == target.shape, f"{target.shape} != {prediction.shape}"
        return prediction, target

    def forward(
            self,
            s: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            ns: torch.Tensor,
            is_done: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            return_td_errors: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the mean squared error loss between the predicted Q-values and the target Q-values.
        The target Q-values are computed based on the Bellman equation, accounting for terminal states.

        Args:
            s: Current states
            a: Actions taken
            r: Rewards received
            ns: Next states
            is_done: Terminal flags
            weights: Importance sampling weights for prioritized replay (optional)
            return_td_errors: If True, return (loss, td_errors) for updating priorities

        Returns:
            loss: Scalar loss value
            td_errors: TD errors for each sample (only if return_td_errors=True)
        """
        assert is_done.dtype is torch.bool, f"Done flags must be dtype 'torch.bool'. Got {is_done.dtype} instead."
        assert a.dim() == 1, f"Actions must be 1-dimensional. Got {a.dim()} instead."
        assert r.dim() == 1, f"Rewards must be 1-dimensional. Got {r.dim()} instead."
        assert is_done.dim() == 1, f"Done flags must be 1-dimensional.¬ Got {is_done.dim()} instead."
        assert s.shape == ns.shape, f"States and next states must have the same shape. Got {s.shape} != {ns.shape}"
        prediction, target = self._fn(s=s, a=a, ns=ns)
        # At the last state use the simplified formula: Q(s,a) = r(s,a) since "s'" doesn't exist.
        target = r + self.gamma * target * (~is_done).to(r.dtype)
        # Compute element-wise TD errors.
        td_errors = prediction - target
        # Apply importance sampling weights if provided
        loss = torch.mean((weights * td_errors) ** 2) if (weights is not None) else torch.mean(td_errors ** 2)
        if return_td_errors: return loss, td_errors.detach()
        return loss
