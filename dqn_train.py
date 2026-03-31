import os
import sys
import warnings
from functools import partial

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.envs import GymWrapper
from torchrl.modules import QValueActor, EGreedyModule, DuelingCnnDQNet, NoisyLinear
from torchrl.data.replay_buffers import LazyMemmapStorage, PrioritizedReplayBuffer
from torchrl.record import CSVLogger

from typing import Callable, Optional

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

if __name__ == "__main__":
    model_space: ModelParameters = ModelParameters(batch_size=128, lr=2e-4, min_lr=1e-8, max_grad_norm=10.)
    paths_space: PathsParameters = PathsParameters()
    names_space: EnvSpaceName = EnvSpaceName()
    # ------------------------------------------
    print(model_space)
    print(paths_space)
    print(names_space)
    # ------------------------------------------
    env_prep = GymPreprocessing(
        partial(AtariPreprocessing, noop_max=0, frame_skip=1, terminal_on_life_loss=False, screen_size=84),
        partial(EpisodicLifeEnv),
        partial(FireResetEnv),
        partial(ClipRewardEnv),
        partial(FrameStackObservation, stack_size=4)
    )
    build_env: Callable[[], GymWrapper] = lambda: create_breakout_env(transform=env_prep)
    envir: GymWrapper = build_env()
    # ------------------------------------------
    logs_config: LogsConfig = LogsConfig(
        paths_space.log_dir, metrics_save_freq=50, weights_save_freq=300, videos_save_freq=300
    )
    logger: SmartLogger = SmartLogger(names_space.actor, options=logs_config, exp_name=paths_space.exp_name)
    # ------------------------------------------
    out_features = envir.action_spec.shape.numel()
    cnn_kwargs = dict(num_cells=(64, 128, 256), kernel_sizes=(8, 4, 3), strides=(4, 2, 1))
    mlp_kwargs = dict(num_cells=256, layer_class=NoisyLinear)
    model: TensorDictModule = Model(
        actor=QValueActor(nn.Sequential(
            Scale(value=255.),
            DuelingCnnDQNet(out_features=out_features, cnn_kwargs=cnn_kwargs, mlp_kwargs=mlp_kwargs)
        ),
            in_keys=[names_space.observation],
            spec=envir.action_spec),
        # Turn off explorer, because we use "NoisyLinear".
        explorer=EGreedyModule(envir.action_spec, 0.0, 0.0, annealing_num_steps=0)
    )
    # ------------------------------------------
    model: TensorDictModule = init_lazy_layers(envir.reset(), model).apply(initialize_weights)
    last_upd: Optional[str] = logger.get_last_update(names_space.actor)
    model.load_state_dict(torch.load(last_upd)) if last_upd else None
    print(f"Weights counts: {n_parameters(model)}")
    # ------------------------------------------
    buffer = PrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        batch_size=model_space.batch_size,
        storage=LazyMemmapStorage(
            max_size=5 * 10 ** 5,
            scratch_dir=paths_space.storage_path,
            existsok=True,
            auto_cleanup=True
        )
    )
    # ------------------------------------------
    collector_kwargs = dict(
        frames_per_batch=200,
        total_frames=-1,
        extend_buffer=False,
        storing_device=model_space.cpu,
        policy_device=model_space.dev
    )
    fill_buffer(init_collector(build_env, **collector_kwargs), buffer, 10 ** 5, show=True)
    # ------------------------------------------
    collector_kwargs = dict(
        frames_per_batch=model_space.batch_size,
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
    optim_method = Optimizer(network=model.to(model_space.dev), action_space=envir.action_spec, params=model_space)
    # ------------------------------------------
    trainer = Trainer(model, optim_method, model_space, names_space, logger, video_maker)
    trainer.train(n_epochs=50000, rb=buffer, loader=collector)
