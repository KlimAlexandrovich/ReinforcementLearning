import numpy as np
import os
from typing import Iterable, Any, Optional, Sequence, Callable, Self
from functools import wraps
from collections import OrderedDict, deque
from dqn_types import ERB, Experience
from SumTree import SumTree
from utils import check_disk_space_for_memmap


class VanillaReplayBuffer(object):
    """
    ReplayBuffer class for storing and sampling experiences in reinforcement learning.
    This class manages a fixed-size buffer to store experience tuples and allows
    for sampling batch data for training reinforcement learning models. It is
    designed to ensure that the buffer does not exceed the defined size by
    discarding old experiences as new ones are added, providing a mechanism to
    shuffle the experiences through random sampling during learning.
    """

    def __init__(self, size: int, iterable: Optional[Iterable] = None):
        if iterable is not None:
            assert all(map(lambda obj: isinstance(obj, Experience), iterable)), \
                "Iterable must contain Experience objects."
        iterable = [] if (iterable is None) else iterable
        self.deque: deque[Experience] = deque(iterable, maxlen=size)

    def __len__(self):
        return len(self.deque)

    def reuse_deque(self, iterable: Iterable) -> Self:
        """ Replaces the entire buffer contents with the provided iterable. """
        assert all(map(lambda obj: isinstance(obj, Experience), iterable)), \
            "Iterable must contain Experience objects."
        self.deque = deque(iterable, maxlen=self.deque.maxlen)
        return self

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray | int | float,
            reward: np.ndarray | int | float,
            next_obs: np.ndarray,
            done: bool
    ) -> None:
        """
        Adds an experience entry to the buffer. This method is responsible for storing
        a single interaction instance consisting of observation, action, reward,
        next observation, and done flag in the buffer for later retrieval or
        processing. It facilitates reinforcement learning training by collecting
        transitional data over episodes.
        """
        data = Experience(obs, action, reward, next_obs, done)
        self.deque.append(data)

    def _encode_sample(
            self, indexes: Iterable[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encodes a sample of experiences from the buffer into separate numpy arrays
        for observations, actions, rewards, next observations, and done flags. This
        method is used to process and structure the data for further usage.
        """
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for idx in indexes:
            exp = self.deque[idx]
            obs.append(exp.obs)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_obs.append(exp.next_obs)
            dones.append(exp.done)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.uint8),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly samples a batch of data from the buffer.
        This method selects a subset of data from the buffer using random sampling without
        replacement based on the specified batch size. The sampled data is encoded into a
        tuple format for further usage.
        """
        indexes = np.random.choice(len(self.deque), size=batch_size)
        return self._encode_sample(indexes=indexes)


def to_numpy(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        replace: Callable[[Any], np.ndarray] = lambda arg: arg if isinstance(arg, np.ndarray) else np.array(arg)
        args = map(replace, args)
        kwargs = {key: replace(value) for key, value in kwargs.items()}
        return func(self, *args, **kwargs)

    return wrapper


class MemMapField:
    def __init__(self, path: str, name: str, dtype: Any, size: int, shape: tuple[int, ...]):
        self.path: str = os.path.join(path, name + ".bin")
        self.name: str = name
        self.dtype: Any = dtype
        self.size: int = size
        self.shape: tuple[int, ...] = tuple(shape)
        self.memmap: Optional[np.memmap] = None
        # Post initialization.
        self.initial_validate()
        self.init_memmap()

    def __repr__(self):
        return f"MemMapField({self.name}, dtype: {self.dtype}, shape: {self.shape})"

    def is_dtype(self) -> bool:
        """ Checks if the provided dtype is valid. """
        try:
            np.dtype(self.dtype)
            return True
        except (TypeError, ValueError):
            return False

    def initial_validate(self) -> bool:
        """ Checks if the field can be initialized successfully. """
        assert isinstance(self.name, str), f"Invalid field name: {self.name}"
        assert all(isinstance(x, int) for x in self.shape), f"Invalid field shape: {self.shape}"
        assert self.is_dtype(), f"Invalid field type: {self.dtype}"
        return True

    def init_memmap(self) -> None:
        """ Initialize memory mapped array. """
        new: bool = not os.path.exists(self.path)
        if new:
            is_enough, details = check_disk_space_for_memmap(self.path, (self.size, *self.shape), np.dtype(self.dtype))
            assert is_enough, f"Not enough disk space to create memmap ({self.path}). Details: {details}"
        mode: str = "w+" if new else "r+"
        self.memmap = np.memmap(self.path, np.dtype(self.dtype), mode, shape=(self.size, *self.shape))
        if new: self.memmap.flush()

    def is_valid(self, value: np.ndarray) -> tuple[bool, dict[str, bool]]:
        """ Checks if the provided value is correct. """
        instance: bool = isinstance(value, np.ndarray)
        dtype: bool = value.dtype == np.dtype(self.dtype)
        shape: bool = value.shape == self.shape
        return instance + dtype + shape, dict(instance=instance, dtype=dtype, shape=shape)

    def add(self, idx: int, value: np.ndarray) -> None:
        """ Add a new sample to the memory-mapped array at the specified index. """
        assert 0 <= idx < self.size, f"Invalid index: {idx}"
        valid, info = self.is_valid(value)
        assert valid, f"Invalid value. Info: {info}"
        self.memmap[idx] = value

    def get(self, idx: Sequence[int]) -> np.ndarray:
        """ Get a sample from the memory-mapped array at the specified index. """
        return self.memmap[idx].copy()


class MemMapDeque(ERB):
    def __init__(self, path: str, max_size: int, fields: Iterable[tuple[str, Any, tuple[int, ...]]]):
        super(MemMapDeque, self).__init__()
        self.storage_dir: str = path
        self.fields: OrderedDict[str, MemMapField] = OrderedDict()
        self.max_size: int = max_size
        # Counters.
        self.counters_path: str = os.path.join(self.storage_dir, "deque_counters.npy")
        self._idx, self._size = None, None
        # Post initialization.
        self._init_memmap(fields), self._init_counters()

    def __repr__(self):
        representation: str = f"MemMapDeque(current idx: {self._idx}, size: {self._size}, max size: {self.max_size}"
        for field in self.fields.values(): representation += f"\n  {field}"
        representation += "\n)"
        return representation

    def __len__(self) -> int:
        return self._size

    def _init_memmap(self, fields: Iterable[tuple[str, Any, tuple[int, ...]]]) -> None:
        """ Initialize memory mapped array. """
        os.makedirs(self.storage_dir, exist_ok=True)
        for data in fields:
            assert len(data) == 3, f"Invalid field definition: {data}"
            name, dtype, shape = data
            self.fields[name] = MemMapField(self.storage_dir, name, dtype, self.max_size, shape)

    def _increment(self) -> None:
        """ Increment the index of the next sample to be written. """
        self._idx = (self._idx + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def _init_counters(self) -> None:
        """ Initialize counters file. """
        if os.path.exists(self.counters_path):
            self._idx, self._size = np.load(self.counters_path, allow_pickle=True)
        else:
            counters: np.ndarray = np.array([0, 0]).astype(np.int64)
            np.save(self.counters_path, counters, allow_pickle=True)
            self._idx, self._size = counters

    def get_counters(self) -> tuple[int, int]:
        """ Get the current counters for the buffer (idx, size). """
        return self._idx, self._size

    def force_save(self) -> None:
        """ Force saving all unsaved samples to disk. """
        counters: np.ndarray = np.array([self._idx, self._size]).astype(np.int64)
        np.save(self.counters_path, counters, allow_pickle=True)
        for field in self.fields.values(): field.memmap.flush()

    @to_numpy
    def add(self, **fields: np.ndarray) -> None:
        """ Add a new sample to the buffer. """
        assert len(fields) == len(self.fields), f"Invalid number of fields: got {len(fields)}."
        for name, ndarray in fields.items():  self.fields[name].add(self._idx, ndarray)
        self._increment()

    def sample(self, size: int) -> dict[str, np.ndarray]:
        """ Sample a batch of experiences from the buffer. """
        assert size <= self._size, f"Requested sample size exceeds the buffer size: {size} > {self._size}"
        indexes: np.ndarray = np.random.choice(self._size, size=size, replace=False)
        sample: dict[str, np.ndarray] = self.sample_indices(indexes)
        return sample

    def sample_indices(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """ Sample specific indices from the buffer. Used for prioritized replay. """
        sample: dict[str, np.ndarray] = {key: field.get(indices) for key, field in self.fields.items()}
        return sample

    def append(self, **fields: np.ndarray) -> None:
        """ Alias for add() to match the standard buffer interface. """
        self.add(**fields)


class PER:
    """
    Prioritized Experience Replay Buffer.
    Samples transition with probability proportional to their TD error.
    Uses important sampling weights to correct for bias.
    Paper: "Prioritized Experience Replay" https://arxiv.org/abs/1511.05952 (Schaul et al., 2015)
    """

    def __init__(
            self,
            buffer: MemMapDeque,
            alpha: float = 0.6,
            beta_start: float = 0.4,
            beta_frames: int = 100000,
            epsilon: float = 1e-6,
    ):
        """
        Args:
            buffer: Base MemMapDeque buffer for storing transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling correction (increases to 1.0)
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant to prevent zero priorities
        """
        self.buffer = buffer
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        # SumTree for efficient priority sampling.
        self.tree = SumTree(capacity=buffer.max_size)
        # Initialize a tree with existing buffer data (if any).
        # Set a default priority for existing transitions.
        self.tree.fill_tree(value=0.0)  # For unused indexes priority equal 0.0.
        for idx in range(len(buffer)): self.tree.update(idx, 1.0)

    @property
    def beta(self) -> float:
        """ Linearly anneal beta from beta_start to 1.0. """
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, **transition: np.ndarray) -> None:
        """ Add transition with maximum priority. """
        # Get max priority from a tree (new samples get the highest priority).
        max_priority = self.tree.get_max_priority()
        # Add to buffer and tree.
        # Why max priority? We don't yet know the "TD-error" of the new experience,
        # so we give it maximum priority to guarantee learning at least once.
        idx, _ = self.buffer.get_counters()
        self.buffer.append(**transition)
        self.tree.update(idx, max_priority)

    def sample(self, batch_size: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        Returns: batch: Dict with transition data
                 indices: Indices of sampled transitions
                 weights: Importance sampling weights
        """
        assert len(self.buffer) >= batch_size, f"Buffer size {len(self.buffer)} < batch size {batch_size}"
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        # Sample proportionally to prioritize (stratified sampling).

        segment = self.tree.total_priority / batch_size
        for step in range(batch_size):
            # Sample uniformly from segment.
            interval = segment * step, segment * (step + 1)
            value = np.random.uniform(*interval)
            # Get the corresponding idx.
            idx = self.tree.get(value)
            indices[step] = idx
            priorities[step] = self.tree.get_priority(idx)

        # Get transitions from buffer.
        batch = self.buffer.sample_indices(indices)
        # Compute importance of sampling weights.
        sampling_probs = priorities / self.tree.total_priority
        weights = (len(self.buffer) * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability.
        self.frame += 1
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        Args: indices: Indices of transitions to update
              td_errors: TD errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def force_save(self):
        """ Save buffer to disk. """
        self.buffer.force_save()
