import numpy as np
import torch
from typing import Sequence, Any
from functools import wraps
from dataclasses import dataclass


class Agent:
    """ Abstract base class for all agents. """

    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated directly.")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """ Evaluates the Q-values for the given states using the Q-network. """

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """ Computes Q-values for the given states using the current model. """

    def sample_actions_by_q_values(self, q_values: np.ndarray, greedy: bool = False) -> np.ndarray:
        """ Selects actions based on Q-values using an epsilon-greedy policy. """

    def sample_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        """ Samples actions based on the provided states and the greediness setting. """


class DataClass:
    """ Abstract base class for all data classes. """

    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated directly.")


class ERB:
    """ Abstract class for defining an experience replay buffer interface. """

    def __len__(self) -> int:
        """ Returns the current size of the buffer. """

    def force_save(self) -> None:
        """ Force saving all unsaved samples to disk. """

    def add(self, **fields: np.ndarray) -> None:
        """ Add a new sample to the buffer. """

    def sample(self, idx: Sequence[int]) -> dict[str, np.ndarray]:
        """ Sample a batch of experiences from the buffer. """


def copy_args[Var](obj: Var) -> Var:
    """
    A decorator function that ensures all arguments and keyword arguments passed to the decorated
    callable are deep copied using NumPy's copy functionality. This prevents modification of the
    original arguments/keyword arguments outside the decorated callable.
    Works both when applied to a function and when applied to a class(wraps it's __init__).
    """
    if isinstance(obj, type):
        orig_init = obj.__init__

        @wraps(orig_init)
        def __init__(self, *args, **kwargs):
            new_args = tuple(np.copy(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
            new_kwargs = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
            orig_init(self, *new_args, **new_kwargs)

        obj.__init__ = __init__
        return obj

    @wraps(obj)
    def wrapper(*args, **kwargs):
        new_args = tuple(np.copy(a) if isinstance(a, np.ndarray) else a for a in args)
        new_kwargs = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
        return obj(*new_args, **new_kwargs)

    return wrapper


@dataclass
@copy_args
class Experience:
    """ Represents a single experience step in a reinforcement learning environment.
        This class is used to encapsulate information about an experience in the
        reinforcement learning paradigm. It contains the observation, action, reward,
        next observation, and a flag indicating whether the episode is done. The purpose
        of this class is to provide a clear and structured way to store and retrieve
        experience data for use in training or evaluation processes. """
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return self.__dict__


@dataclass
class EnvSpaceName:
    """ DataClass contains terms of literal strings.
        This class is used to encapsulate information about an environment space.
        Warning: this class is compatible with another module of this package.
        If you change some "logic", check compatibility. """
    actor: str = "actor"
    explorer: str = "explorer"
    action: str = "action"
    observation: str = "observation"
    reward: str = "reward"
    terminated: str = "terminated"
    truncated: str = "truncated"
    done: str = "done"
    next: str = "next"
    loss: str = "loss"
    td_error: str = "td_error"
    index: str = "index"
    transformed: str = "transformed"


@dataclass
class ModelParameters:
    # Input data parameters.
    n_frames: int = 4
    # Train parameters.
    lr: float = 2e-2
    min_lr: float = 1e-5
    max_grad_norm: float = 10.
    batch_size: int = 64
    soft_update_eps: float = 0.995
    n_epochs: int = 2 * 10 ** 5
    # Exploration.
    rb_expansion: int = 10
    # Devices.
    dev: torch.device = torch.device("mps:0")
    cpu: torch.device = torch.device("cpu:0")


@dataclass
class PathsParameters:
    """ DataClass contains paths and devices. """
    exp_name: str = "experience"
    log_dir: str = "breakout_logs"
    storage_path: str = f"{log_dir}/dump/storage"
