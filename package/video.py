import os
import numpy as np
import time
import cv2
import plotly.graph_objects as go
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record import CSVLogger, VideoRecorder
from utils import get_last_update
from typing import Any, Optional


def unstack_frames(tensordict: TensorDict, key: str = "observation", flip: bool = False) -> np.ndarray:
    """
    Computes the mean of stacked frames to create a single-channel observation for visualization.

    Args:
        tensordict: The TensorDict containing environment observations.
        key: The key for observations in the TensorDict.
        flip: Whether to flip the frames vertically.

    Returns:
        A numpy array of the processed frames.
    """
    obs: torch.Tensor = tensordict[key].float().cpu()
    assert obs.dim() == 4, f"Function awaiting observation with 4 dimensions, got {obs.dim()}."
    mean_frames: np.ndarray = obs.mean(dim=1).numpy().reshape(obs.size(0), obs.size(2), obs.size(3), 1)
    if flip:
        for idx, frame in enumerate(mean_frames):
            mean_frames[idx] = np.flipud(frame)
    return mean_frames


class Recorder:
    """
    Handles recording agent interactions within the environment.

    Utilizes torchrl's VideoRecorder to capture and save gameplay videos.
    """

    def __init__(self, manager: CSVLogger, env: GymWrapper, deterministic: bool = True):
        """
        Initializes the Recorder.

        Args:
            manager: The logger to manage video storage.
            env: The environment to record.
            deterministic: Whether to use deterministic policy during recording.
        """
        self.rec = VideoRecorder(manager, tag="replay", options=dict(crf="1", preset="slow"), fps=30)
        self.env = TransformedEnv(env, self.rec)
        self.deterministic = ExplorationType.DETERMINISTIC if deterministic else None
        self.__kwargs = dict(break_when_any_done=False, auto_reset=True, auto_cast_to_device=True)

    def __call__(self, max_steps: int, policy: Optional[TensorDictBase] = None):
        """
        Executes a rollout and records the video.

        Args:
            max_steps: Maximum number of steps to record.
            policy: The policy to use for action selection.
        """
        with set_exploration_type(self.deterministic):
            self.env.rollout(max_steps, policy, **self.__kwargs)
        self.rec.dump(step=int(time.time()))

    def last_rec(self) -> str:
        """
        Returns the path to the most recent recording.

        Returns:
            The file path to the last video record.
        """
        directory = os.path.join(self.rec.logger.log_dir, self.rec.logger.exp_name, "videos")
        return get_last_update(directory)

    def shoot(self, max_steps: int, policy: Optional[TensorDictBase] = None):
        """
        Triggers a recording session.

        Args:
            max_steps: Maximum number of steps to record.
            policy: The policy to use for action selection.
        """
        self.__call__(max_steps, policy)


def show_video(path: str, fps: int = 30) -> None:
    """
    Displays a video file in a window using OpenCV.

    Args:
        path: Path to the video file.
        fps: Frames per second for playback.
    """
    capture = cv2.VideoCapture(path)
    # Read and display video frames
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret: break
        cv2.imshow("Video", frame)
        time.sleep(1.0 / fps)
        # Press "q" to exit.
        if cv2.waitKey(25) & 0xFF == ord("q"): break
    capture.release()
    cv2.destroyAllWindows()


class VideoPlayer:
    """
    An interactive video player built with Plotly for Jupyter Notebooks.

    Supports both RGB and grayscale frame sequences.
    """

    def __init__(self, frames: np.ndarray, fps: int = 20, title: str = "Application"):
        """
        Initialize the video player for visualizing a sequence of frames.

        :param frames: np.ndarray of shape (n_frames, height, width, channels).
        :param fps: Frames per second.
        :param title: Title of the visualization.
        """
        assert isinstance(frames, np.ndarray), f"Incorrect frames dtype, should be 'np.ndarray', not {type(frames)}."
        assert frames.ndim == 4, f"Frames dim = {frames.ndim} ≠ 4."

        self.frames: np.ndarray = frames
        self.n_frames: int = self.frames.shape[0]
        self.is_rgb: bool = self.frames.shape[-1] == 3

        # Data preparation: remove the channel dimension for Heatmap if the image is grayscale.
        self.frames = self.frames if self.is_rgb else self.frames.squeeze(-1)

        # Display settings.
        self.fps: int = fps
        self.frame_duration: int = 1000 // fps
        self.title: str = title

    def render(self, frame_idx: int) -> go.Image | go.Heatmap:
        """
        Render a single frame based on the data type (RGB or grayscale).

        :param frame_idx: Index of the frame to render.
        :return: go.Image for RGB or go.Heatmap for grayscale.
        """
        frame: np.ndarray = self.frames[frame_idx]
        if self.is_rgb:
            return go.Image(z=frame)
        return go.Heatmap(z=frame, colorscale="Gray", showscale=False)

    @staticmethod
    def animations_settings(
            frame_duration: int,
            method_on_click: Any = None,
            redraw: bool = True,
            immediate: bool = False,
            from_current: bool = True,
    ) -> list[Any]:
        """
        Universal configuration for Plotly animation settings.

        :param frame_duration: Duration of the frame in milliseconds.
        :param method_on_click: Plotly animation method argument (e.g., None for all frames, [None] for pause).
        :param redraw: Whether to redraw the entire plot (important for Heatmap/Image updates).
        :param immediate: If True, the animation will start immediately, interrupting current one.
        :param from_current: If True, the animation starts from the current frame.
        :return: A list containing the method argument and the configuration dictionary.
        """
        args = {
            "frame": {"duration": frame_duration, "redraw": redraw},
            "fromcurrent": from_current,
            "transition": {"duration": 0}
        }
        if immediate:
            args["mode"] = "immediate"

        return [method_on_click, args]

    def _create_btn(self, label: str, method_on_click: Any, frame_duration: int, **kwargs) -> dict:
        """
        Helper method to create a single control button.

        :param label: Button label text.
        :param method_on_click: Argument for the 'animate' method.
        :param frame_duration: Duration for the animation step.
        :param kwargs: Additional arguments for animations_settings.
        :return: A dictionary representing a Plotly button.
        """
        return dict(
            label=label,
            method="animate",
            args=self.animations_settings(frame_duration, method_on_click, **kwargs)
        )

    def _create_buttons(self) -> list[dict]:
        """
        Create the control button block (Play/Pause).

        :return: A list of dictionaries for Plotly updatemenus.
        """
        play_btn = self._create_btn(
            label="▶ Play",
            method_on_click=None,  # None means play the entire sequence
            frame_duration=self.frame_duration,
            from_current=True
        )

        pause_btn = self._create_btn(
            label="⏸ Pause",
            method_on_click=[None],  # [None] interrupts the current animation
            frame_duration=0,
            immediate=True,
            redraw=False
        )

        return [dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0, xanchor="right", yanchor="top",
            buttons=[play_btn, pause_btn]
        )]

    def _create_slider_step(self, i: int) -> dict:
        """
        Create a single step for the interactive slider.

        :param i: Index of the frame.
        :return: A dictionary representing a slider step.
        """
        return dict(
            method="animate",
            label=str(i),
            args=self.animations_settings(
                frame_duration=self.frame_duration,
                method_on_click=[str(i)],  # Jump to a specific named frame
                immediate=True,
                redraw=True
            )
        )

    def _create_slider(self) -> list[dict]:
        """
        Create the interactive slider for manual frame navigation.

        :return: A list of dictionaries for Plotly sliders.
        """
        return [dict(
            active=0,
            x=0.15, y=0, len=0.85,
            xanchor="left", yanchor="top",
            currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor="right"),
            steps=[self._create_slider_step(i) for i in range(self.n_frames)]
        )]

    def plot(self, width: int = 600, height: int = 600) -> go.Figure:
        """
        Assemble the final Plotly figure.

        :param width: Width of the figure in pixels.
        :param height: Height of the figure in pixels.
        :return: A go.Figure object.
        """
        figure = go.Figure(
            data=[self.render(0)],
            layout=go.Layout(
                title=self.title,
                width=width,
                height=height,
                updatemenus=self._create_buttons(),
                sliders=self._create_slider(),
                margin=dict(l=20, r=20, t=50, b=50),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            ),
            frames=[go.Frame(data=[self.render(i)], name=str(i)) for i in range(self.n_frames)]
        )
        return figure

    def show(self, **kwargs) -> None:
        """
        Display the player in a notebook or browser.

        :param kwargs: Arguments passed to the plot method (width, height).
        """
        self.plot(**kwargs).show()
