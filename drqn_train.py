import os
import sys
import warnings
from functools import partial

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.envs import GymWrapper, Compose, InitTracker, StepCounter, TransformedEnv
from torchrl.modules import QValueActor, ConvNet, EGreedyModule, LSTMModule, MLP
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.record import CSVLogger

from typing import Callable, Optional, Any

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv, ClipRewardEnv

warnings.filterwarnings("ignore")
gym.register_envs(ale_py)
if os.path.abspath("package") not in sys.path: sys.path.append(os.path.abspath("package"))

from package.environment import GymPreprocessing, create_breakout_env
from package.dqn_types import ModelParameters, PathsParameters, EnvSpaceName
from package.utils import fill_buffer, init_collector
from package.video import Recorder
from package.Logger import SmartLogger, LogsConfig
from package.modules import (Model,
                             Scale,
                             Optimizer,
                             Trainer,
                             initialize_weights,
                             init_lazy_layers,
                             n_parameters)


class DRQN(Model):
    def __init__(self, cnn: dict[str, Any], fc: dict[str, Any], hidden_size: int):
        # ----------- Keys -----------
        obs_key: str = "observation"
        scale_key: str = "scales"
        rnn_key: str = "embed"
        mlp_key: str = "action_value"
        # ----------- Modules -----------
        scale = TensorDictModule(Scale(value=255.), in_keys=obs_key, out_keys=scale_key)
        backbone = TensorDictModule(ConvNet(**cnn), in_keys=scale_key, out_keys=rnn_key)
        memory = LSTMModule(input_size=hidden_size * 2 * 2, hidden_size=hidden_size, in_key=rnn_key, out_key=rnn_key)
        mlp = TensorDictModule(MLP(**fc), in_keys=rnn_key, out_keys=mlp_key)
        super().__init__(scale=scale, backbone=backbone, memory=memory, mlp=mlp)


if __name__ == "__main__":
    mean_episode_len: int = 32
    model_space: ModelParameters = ModelParameters(
        n_frames=4,
        n_epochs=10 ** 4,
        batch_size=8 * mean_episode_len,
        rb_expansion=64,
        lr=1e-4,
        min_lr=1e-5,
        weight_decay=1e-6,
        max_grad_norm=1.,
        soft_update_eps=0.999
    )
    paths_space: PathsParameters = PathsParameters(exp_name="drqn", log_dir="breakout_logs")
    names_space: EnvSpaceName = EnvSpaceName()
    # ------------------------------------------
    print(model_space)
    print(paths_space)
    print(names_space)
    # ------------------------------------------
    # In our preprocessing, we use frame_skip=4 (Atari standard).
    # Higher values like 5-7 make the ball movement too choppy for the model to track.
    env_prep = GymPreprocessing(
        partial(AtariPreprocessing, noop_max=20, frame_skip=4, terminal_on_life_loss=False, screen_size=48),
        # EpisodicLifeEnv provides the "end episode on life loss" signal for faster training,
        # while AtariPreprocessing(terminal_on_life_loss=False) ensures reset() behaves correctly.
        partial(EpisodicLifeEnv),
        partial(FireResetEnv),
        partial(ClipRewardEnv),
        partial(FrameStackObservation, stack_size=model_space.n_frames)
    )
    build_env: Callable[[], GymWrapper] = lambda: TransformedEnv(
        create_breakout_env(transform=env_prep), Compose(transforms=(StepCounter(), InitTracker()))
    )
    envir: GymWrapper = build_env()
    # ------------------------------------------
    logs_config: LogsConfig = LogsConfig(
        log_dir=paths_space.log_dir,
        metrics_save_freq=60,
        weights_save_freq=300,
        videos_save_freq=300
    )
    logger: SmartLogger = SmartLogger(names_space.actor, options=logs_config, exp_name=paths_space.exp_name)
    # ------------------------------------------
    latent_size: int = 256
    mlp_kwargs: dict[str, Any] = dict(out_features=envir.action_spec.shape.numel(), num_cells=(latent_size,))
    cnn_kwargs: dict[str, Any] = dict(
        num_cells=(latent_size // 4, latent_size // 2, latent_size),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1)
    )
    model: TensorDictModule = Model(
        actor=QValueActor(
            DRQN(cnn=cnn_kwargs, fc=mlp_kwargs, hidden_size=latent_size),
            in_keys=names_space.observation,
            spec=envir.action_spec
        ),
        explorer=EGreedyModule(
            spec=envir.action_spec,
            # Starting with high exploration to discover initial rewards (breaking the first brick).
            eps_init=1.,
            eps_end=0.01,
            annealing_num_steps=model_space.n_epochs // 2  # Gradual decay over half the training epochs.
        )
    )
    # ------------------------------------------
    model: TensorDictModule = init_lazy_layers(envir.reset(), model).apply(initialize_weights)
    last_upd: Optional[str] = logger.get_last_update(names_space.actor)
    model.load_state_dict(torch.load(last_upd)) if last_upd else None
    print(f"Weights counts: {n_parameters(model)}")
    # ------------------------------------------
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            max_size=2.5 * 10 ** 5,
            scratch_dir=paths_space.storage_path,
            existsok=True,
            auto_cleanup=True
        ),
        sampler=SliceSampler(
            slice_len=mean_episode_len,  # Mean episode len.
            end_key=("next", "done"),
            traj_key=("collector", "traj_ids"),
            cache_values=True,
            strict_length=False
        ),
        batch_size=model_space.batch_size,
        prefetch=10
    )
    # ------------------------------------------
    collector_kwargs = dict(
        frames_per_batch=50,
        total_frames=-1,
        storing_device=model_space.cpu,
        policy_device=model_space.dev,
        extend_buffer=False
    )
    _ = fill_buffer(init_collector(build_env, model, **collector_kwargs), buffer, 5 * 10 ** 4, show=True)
    # ------------------------------------------
    collector_kwargs = dict(
        frames_per_batch=model_space.batch_size // 2,
        total_frames=-1,
        extend_buffer=False,
        storing_device=model_space.cpu,
        policy_device=model_space.dev
    )
    # ------------------------------------------
    video_maker = Recorder(
        CSVLogger(paths_space.exp_name, paths_space.log_dir, video_format="mp4", video_fps=30),
        build_env(),
        deterministic=True
    )
    collector = init_collector(build_env, model, **collector_kwargs)
    optim_method = Optimizer(
        network=model.to(model_space.dev),
        action_space=envir.action_spec,
        params=model_space
    )
    # ------------------------------------------
    trainer = Trainer(model, optim_method, model_space, names_space, logger, video_maker)
    trainer.train(n_epochs=model_space.n_epochs, rb=buffer, loader=collector)
