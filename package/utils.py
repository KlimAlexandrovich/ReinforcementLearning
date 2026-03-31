import torch
from torchrl.collectors import Collector
from torchrl.envs import GymWrapper
from torchrl.data import ReplayBuffer
from tensordict.nn import TensorDictModule
import cv2
import numpy as np
import gymnasium as gym
import time
from tqdm import tqdm
import pickle
import os
import shutil
import json
from typing import Optional, Callable, Any
# Package.
from Samplers import ActionSampler
from dqn_types import DataClass, Agent, ERB


def is_strict_json(obj: object) -> bool:
    """ Check if the object can be converted to JSON. """
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def write_params(args: DataClass, current: Optional[dict] = None) -> dict:
    """ Write parameters from a DataClass object into a dictionary. """
    # TODO: Add support for nested DataClass objects.
    current = dict() if (current is None) else current
    for key, value in args.__dict__.items():
        branch = dict()
        new_value = value if not isinstance(value, DataClass) else {value.__class__.__name__: branch}
        new_value = new_value if is_strict_json(new_value) else str(new_value)
        current[key] = new_value
        # The dictionary is a mutable object, so we will be making changes inplace.
        if isinstance(value, DataClass): _ = write_params(value, branch)
    return current


def write_json(path: str, data: Any) -> None:
    """
    Writes data to a JSON file.

    Args:
        path: The file path to write to.
        data: The data to serialize.
    """
    assert is_strict_json(data), f"Object {data} cannot be converted to JSON."
    with open(path, "w", encoding="utf-8") as file: json.dump(data, file, indent=4)


def read_json(path: str) -> Any:
    """
    Reads data from a JSON file.

    Args:
        path: The file path to read from.

    Returns:
        The deserialized data.
    """
    with open(path, "r", encoding="utf-8") as file: return json.load(file)


def check_disk_space_for_memmap(path: str, shape: tuple[int, ...], dtype: Any) -> tuple[bool, dict]:
    """ Checks if there is enough free disk space to create a memory mapped array. """
    num_elements: int = np.prod(shape, dtype=np.int64)
    item_size: int | float = np.dtype(dtype).itemsize
    required_bytes: int | float = num_elements * item_size

    target_dir = os.path.dirname(os.path.abspath(path))
    # Ascends a path until the existing directory is found.
    if not os.path.exists(target_dir):
        while not os.path.exists(target_dir) and target_dir != os.path.dirname(target_dir):
            target_dir = os.path.dirname(target_dir)

    free_bytes: float = shutil.disk_usage(target_dir).free
    is_enough: bool = free_bytes > required_bytes

    details = {
        "required_gb": required_bytes / (1024 ** 3),
        "free_gb": free_bytes / (1024 ** 3),
        "diff_gb": (free_bytes - required_bytes) / (1024 ** 3),
        "item_size_bytes": item_size
    }

    return is_enough, details


def get_last_update(directory: str) -> Optional[str]:
    """
    Finds and returns the path to the most recently modified file in a directory.

    Args:
        directory: The directory to search in.

    Returns:
        The path to the newest file, or None if the directory is empty.
    """
    assert os.path.isdir(directory), f"Directory {directory} does not exist."
    listdir: list[str] = os.listdir(directory)
    abs_listdir: list[str] = [os.path.join(directory, path) for path in listdir]
    # "os.path.isfile" left directories if an absolute path exists.
    files: list[str] = list(filter(os.path.isfile, abs_listdir))
    if len(files) == 0: return None
    # Return the newest file.
    last: str = max(files, key=os.path.getmtime)
    return last


def catch_exception(func: Callable) -> Callable:
    """ Decorator to catch exceptions raised by a function. """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            print(f"Function: {func.__name__}\nArgs: {args}\nKwargs: {kwargs}\nError: {error}")

    return wrapper


def except_keyboard_interrupt(func):
    """
    Decorator to catch KeyboardInterrupt and print a friendly message.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("Process interrupted by user.")

    return wrapper


@catch_exception
def pickle_serialize(obj: Any, filename: str) -> None:
    with open(filename, "wb") as file: pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    return None


@catch_exception
def pickle_deserialize(filename: str) -> Any:
    with open(filename, "rb") as file: obj = pickle.load(file)
    return obj


def merge_frame_stack(obs: np.ndarray) -> np.ndarray:
    """ A helper function to plot a frame stack as a single human-interpretable image.
        Brighter pixels are more recent, pale pixels are older.
        Motions go from pale to bright. """
    weights = np.ones(obs.shape[0], dtype=float)
    weights[-1] += weights.sum()
    weights /= weights.sum()
    result = (weights.reshape(-1, 1, 1) * obs).sum(0)
    return result


def show_game(env: gym.Env, sampler: ActionSampler, n_frames: int = 100, freq: float = 0.01) -> None:
    """ Displays the game session. """  # TODO: Add sampler.
    win_name = "To close press 'q'."
    obs, _ = env.reset()
    for i in range(n_frames):
        cv2.imshow(win_name, env.render())
        action = sampler(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: obs, _ = env.reset()
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        time.sleep(freq)
    cv2.destroyWindow(win_name)


def evaluate(
        env: gym.Env,
        agent: Agent,
        n_games: int = 1,
        greedy: bool = False,
        frames_max: int = 10000,
        seed: Optional[int] = None
) -> float:
    """
    Evaluates the performance of a trained DQN agent in a given environment over a specified
    number of games. The function executes episodes of the environment, collects rewards,
    and computes the average reward over the episodes as a metric of performance.
    The evaluation process can be performed using either a greedy or an exploratory policy.
    """
    rewards_history = []
    for _ in range(n_games):
        obs, _ = env.reset(seed=seed)
        total_game_reward = 0.
        # Executes an episode; accumulates reward until termination or truncation.
        for _ in range(frames_max):
            obs = np.expand_dims(np.array(obs), axis=0)
            action = agent.sample_actions(states=obs, greedy=greedy).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_game_reward += reward
            if terminated or truncated: break
        rewards_history.append(total_game_reward)
    return float(np.mean(rewards_history))


@torch.no_grad()
def play_and_record(
        env: gym.Env,
        sampler: ActionSampler,
        buffer_obj: ERB,
        n_steps: int = 1,
        show: bool = False
) -> float:
    """
    Play the game for exactly "n_steps", record every (s,a,r,s', done) to replay buffer.
    Whenever the game ends due to "termination" or "truncation", add a record with done=terminated and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.
    """
    obs, _ = env.reset()
    sum_rewards = 0.
    bar = tqdm(range(n_steps), desc="Playing game...") if show else range(n_steps)
    for _ in bar:
        action = sampler(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer_obj.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=terminated)
        sum_rewards += reward
        obs = next_obs
        if terminated or truncated: obs, _ = env.reset()
    bar.set_description("Game over.") if show else None
    return sum_rewards


def init_collector(
        builder: Callable[[], GymWrapper], network: Optional[TensorDictModule] = None, **kwargs) -> Collector:
    """
    Initializes a torchrl Collector for environment interaction.

    Args:
        builder: A function that creates a new environment instance.
        network: The policy network used for action selection.
        **kwargs: Additional arguments for the Collector.

    Returns:
        An initialized Collector instance.
    """
    return Collector(create_env_fn=builder, policy=network, **kwargs)


@except_keyboard_interrupt
@torch.no_grad()
def fill_buffer(loader: Collector, rb: ReplayBuffer, size: int, show: bool = False) -> None:
    """
    Fills the replay buffer with initial experience collected from the environment.

    Args:
        loader: The collector used to gather frames.
        rb: The replay buffer to store the experience.
        size: The number of frames to collect.
        show: Whether to display a progress bar.
    """
    assert loader.extend_buffer is False, "Implemented only for manual buffer extending!"
    n_iters = size // loader.frames_per_batch
    progress_bar = tqdm(enumerate(loader), total=n_iters) if show else enumerate(loader)
    for iteration, batch in progress_bar:
        rb.extend(batch)
        n_iters -= 1
        if n_iters < 0: break
