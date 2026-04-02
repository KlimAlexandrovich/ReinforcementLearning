import os
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable, Optional

import gymnasium as gym
import ale_py

warnings.filterwarnings("ignore")
gym.register_envs(ale_py)

from stable_baselines3.common.base_class import BaseAlgorithm, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from .Logger import SmartLogger


def is_notebook() -> bool:
    """
    Checks if the code is running in an IPython notebook environment.

    :return: True if running in a ZMQ-based IPython shell (e.g., Jupyter), False otherwise.
    """
    try:
        import IPython
        shell = IPython.get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except (NameError, ImportError):
        return False


class Support(ABC):
    """
    Abstract base class for assistant services used during training.
    """

    @abstractmethod
    def freq(self) -> int:
        """
        Returns the frequency (in steps) at which the assistant should be called.

        :return: Call frequency.
        """
        pass

    @abstractmethod
    def __call__(self) -> dict[str, int | float | bool | str]:
        """
        Executes the assistant's primary functionality.

        :return: A dictionary of metrics or results to be logged.
        """
        pass


class EvaluateReward(Support):
    """
    Assistant for evaluating the model's performance by calculating mean reward over episodes.
    """

    def __init__(
            self,
            network: BaseAlgorithm,
            env: gym.Env | VecEnv,
            freq: int = 1,
            n_episodes: int = 10,
            deterministic: bool = False
    ):
        """
        Initializes the reward evaluator.

        :param network: The RL model to evaluate.
        :param env: The environment used for evaluation.
        :param freq: Evaluation frequency (in steps).
        :param n_episodes: Number of episodes to run for evaluation.
        :param deterministic: Whether to use deterministic actions.
        """
        self.network: BaseAlgorithm = network
        self.env: gym.Env = env
        self._freq: int = freq
        self.n_episodes: int = n_episodes
        self.deterministic: bool = deterministic

    def freq(self) -> int:
        """
        Returns the evaluation frequency.

        :return: Frequency as integer.
        """
        return self._freq

    def __call__(self) -> dict[str, int | float]:
        """
        Performs policy evaluation and returns the results.

        :return: Dictionary containing 'mean_reward' and 'std_reward'.
        """
        mean_reward, std_reward = evaluate_policy(self.network, self.env, self.n_episodes, self.deterministic)
        return dict(mean_reward=mean_reward, std_reward=std_reward)


class Checkpointer(Support):
    """
    Assistant for saving model checkpoints periodically.
    """

    def __init__(self, network: BaseAlgorithm, freq: int = 1, path: str = "./models"):
        """
        Initializes the checkpointer.

        :param network: The RL model to save.
        :param freq: Checkpoint frequency (in steps).
        :param path: Directory where models will be saved.
        """
        self.network: BaseAlgorithm = network
        self.path: str = path
        self._freq: int = freq

    def freq(self) -> int:
        """
        Returns the checkpoint frequency.

        :return: Frequency as integer.
        """
        return self._freq

    def __call__(self) -> dict[str, str | bool]:
        """
        Saves the current model and returns the save path.

        :return: Dictionary containing 'checkpoint' path and 'success' status.
        """
        path: str = os.path.join(self.path, f"model_{time.time()}")
        self.network.save(path)
        return dict(checkpoint=path, success=True)


class VideoWriter(Support):
    """
    Assistant for recording gameplay videos of the agent.
    """

    def __init__(
            self,
            network: BaseAlgorithm,
            builder: Callable[[], gym.Env],
            freq: int = 1,
            duration: int = 500,
            fps: int = 30,
            path: str = "./videos"
    ):
        """
        Initializes the video writer.

        :param network: The RL model to record.
        :param builder: Function that creates a fresh environment for recording.
        :param freq: Recording frequency (in steps).
        :param duration: Number of steps to record per video.
        :param path: Directory where videos will be saved.
        """
        self.network: BaseAlgorithm = network
        self.builder = builder
        self.path: str = path
        self.duration: int = duration
        self.fps: int = fps
        self._freq: int = freq
        self.initial_step()

    def initial_step(self):
        os.makedirs(self.path, exist_ok=True)

    def freq(self) -> int:
        """
        Returns the recording frequency.

        :return: Frequency as integer.
        """
        return self._freq

    def record(self, path: str) -> None:
        """
        Plays and records it to a file.
        """
        # ----- Create a fresh environment -----
        environment: gym.Env = self.builder()
        observation, _ = environment.reset()
        # ----- Get the initial frame size -----
        frame: np.ndarray = environment.render()
        height, width, _ = frame.shape
        # ----- Initialize the video writer -----
        assert "VideoWriter_fourcc" in dir(cv2), "Please install opencv-python to use VideoWriter."
        fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, self.fps, (width, height))
        # ----- Play the game and record the video -----
        for _ in range(self.duration):
            # ----- Convert RGB to BGR and write to frame -----
            frame_bgr = cv2.cvtColor(environment.render(), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            # ----- Make prediction and step -----
            action, _ = self.network.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated: observation, _ = environment.reset()
        # ----- Release the video writer -----
        out.release()
        environment.close()

    def __call__(self) -> dict[str, str | bool]:
        """
        Triggers the recording process and returns the status.

        :return: Dictionary containing 'checkpoint' (video path) and 'success' status.
        """
        path: str = os.path.join(self.path, f"replay_{int(time.time())}.mp4")
        self.record(path)
        return dict(checkpoint=path, success=True)


class Callback(BaseCallback):
    """
    Custom callback for monitoring and controlling the PPO training process.
    Integrates multiple 'Support' assistants and logging via SmartLogger.
    """

    def __init__(
            self,
            *assistants: Support,
            stop_criterion: Optional[Support] = None,
            writer: Optional[SmartLogger] = None,
            verbose: int = 0,
            show_progress: int = 0,
    ):
        """
        Initializes the training callback.

        :param assistants: Variable number of Support assistants (evaluators, checkpointers, etc.).
        :param stop_criterion: Optional assistant that determines when to stop training.
        :param writer: Optional SmartLogger instance for logging metrics.
        :param verbose: Verbosity level for BaseCallback.
        :param show_progress: Total timesteps to display in the progress bar.
        """
        super().__init__(verbose)
        self.assistants: tuple[Support] = assistants
        self.stop_criterion: Optional[Support] = stop_criterion
        self.writer: Optional[SmartLogger] = writer
        self.initial_state: bool = True
        self._bar: Optional[tqdm] = tqdm(total=show_progress, desc="Training") if show_progress else None

    def _on_step(self) -> bool:
        """
        Method called by the model at each step during a rollout collection.

        :return: False if training should be aborted, True otherwise.
        """
        if self._bar is not None:
            self._bar.update()
        # ----------------------------
        if self.initial_state:
            self.initial_state = False
            return True
        # ----------------------------
        for assistant in self.assistants:
            if self.n_calls % assistant.freq() != 0: continue
            for key, value in assistant().items():
                self.logger.record(key, value)
        # ----------------------------
        if self.writer is not None:
            if self.n_calls % self.writer.options.metrics_save_freq == 0:
                for key, value in self.logger.name_to_value.items():
                    if isinstance(key, str): key = key.replace("/", "_")
                    self.writer.set_scalar(time.time(), key, value)
        # ----------------------------
        if self.stop_criterion is not None:
            criterion: dict[str, int | float] = self.stop_criterion()
            if all(criterion.values()):
                return False
        return True
