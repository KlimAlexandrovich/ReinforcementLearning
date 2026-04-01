import os
import sys
import warnings
from functools import partial

import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import GymWrapper
from torchrl.modules import QValueActor, EGreedyModule, DuelingCnnDQNet
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
    model_space: ModelParameters = ModelParameters(
        n_frames=4,
        n_epochs=2 * 10 ** 5,
        batch_size=64,
        rb_expansion=64,
        lr=1e-4,
        min_lr=1e-5,
        weight_decay=1e-5,
        max_grad_norm=1.,
        soft_update_eps=0.995
    )
    paths_space: PathsParameters = PathsParameters(exp_name="dqn", log_dir="breakout_logs")
    names_space: EnvSpaceName = EnvSpaceName()
    # ------------------------------------------
    print(model_space)
    print(paths_space)
    print(names_space)
    # ------------------------------------------
    env_prep = GymPreprocessing(
        partial(AtariPreprocessing, noop_max=20, frame_skip=4, terminal_on_life_loss=False, screen_size=84),
        partial(EpisodicLifeEnv),
        partial(FireResetEnv),
        partial(ClipRewardEnv),
        partial(FrameStackObservation, stack_size=model_space.n_frames)
    )
    build_env: Callable[[], GymWrapper] = lambda: create_breakout_env(transform=env_prep)
    envir: GymWrapper = build_env()
    # ------------------------------------------
    logs_config: LogsConfig = LogsConfig(
        log_dir=paths_space.log_dir,
        metrics_save_freq=50,
        weights_save_freq=500,
        videos_save_freq=500
    )
    logger: SmartLogger = SmartLogger(names_space.actor, options=logs_config, exp_name=paths_space.exp_name)
    # ------------------------------------------
    action_spec = envir.action_spec.shape.numel()
    cnn_kwargs = dict(num_cells=(32, 64, 128), kernel_sizes=(8, 4, 3), strides=(4, 2, 1))
    mlp_kwargs = dict(num_cells=512)
    model: TensorDictModule = Model(
        scale=TensorDictModule(
            Scale(value=255.),
            in_keys=names_space.observation,
            out_keys=names_space.transformed
        ),
        actor=QValueActor(
            DuelingCnnDQNet(
                out_features=action_spec,
                cnn_kwargs=cnn_kwargs,
                mlp_kwargs=mlp_kwargs
            ),
            in_keys=[names_space.transformed],
            spec=envir.action_spec
        ),
        explorer=EGreedyModule(
            spec=envir.action_spec,
            eps_init=0.7,  # When the learning process was resumed, the initial epsilon decreased from 1 to x.
            eps_end=0.01,
            annealing_num_steps=model_space.n_epochs // 2
        )
    )
    # ------------------------------------------
    model: TensorDictModule = init_lazy_layers(envir.reset(), model).apply(initialize_weights)
    last_upd: Optional[str] = logger.get_last_update(names_space.actor)
    model.load_state_dict(torch.load(last_upd)) if last_upd else None
    print(f"Weights counts: {n_parameters(model)}")
    # ------------------------------------------
    buffer = PrioritizedReplayBuffer(
        alpha=0.6,
        beta=0.5,
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
        frames_per_batch=model_space.batch_size,
        total_frames=-1,
        extend_buffer=False,
        storing_device=model_space.cpu,
        policy_device=model_space.dev
    )
    _ = fill_buffer(init_collector(build_env, model, **collector_kwargs), buffer, 10 ** 5, show=True)
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
    optim_method = Optimizer(
        network=model.to(model_space.dev),
        action_space=envir.action_spec,
        params=model_space
    )
    # ------------------------------------------
    trainer = Trainer(model, optim_method, model_space, names_space, logger, video_maker)
    trainer.train(n_epochs=model_space.n_epochs, rb=buffer, loader=collector)
