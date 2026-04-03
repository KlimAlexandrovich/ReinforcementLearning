import os
import sys
import warnings
from functools import partial
from typing import Callable, Optional

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv, ClipRewardEnv
from stable_baselines3.common.utils import ConstantSchedule

warnings.filterwarnings("ignore")
gym.register_envs(ale_py)
# Add project root to sys.path to allow imports from 'package'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from package.environment import GymPreprocessing, create_breakout_env_gym
from package.dqn_types import ModelParameters, PathsParameters
from package.Logger import SmartLogger, LogsConfig
from package.sb3_utils import Callback, Checkpointer, EvaluateReward, VideoWriter, Support, is_notebook

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

if __name__ == "__main__":
    # ----------------- Setup environment -----------------
    env_prep = GymPreprocessing(
        partial(AtariPreprocessing, noop_max=20, frame_skip=4, screen_size=64),
        partial(EpisodicLifeEnv),
        partial(FireResetEnv),
        partial(ClipRewardEnv),
        partial(FrameStackObservation, stack_size=4)
    )
    build_env: Callable[[], gym.Env] = lambda: create_breakout_env_gym(transform=env_prep)
    dummy_env: Callable[[int], DummyVecEnv] = lambda cnt: DummyVecEnv(list(build_env for _ in range(cnt)))
    multi_env: Callable[[int], SubprocVecEnv] = lambda cnt: SubprocVecEnv(list(build_env for _ in range(cnt)))
    # ----------------- Model -----------------
    model_space: ModelParameters = ModelParameters(
        lr=2.5e-4,
        batch_size=512,
        n_epochs=10,
        n_steps=512,
        n_parallel=8,
        max_grad_norm=0.5,
        n_frames=4
    )
    print(model_space)
    envir = dummy_env(model_space.n_parallel) if is_notebook() else multi_env(model_space.n_parallel)
    model = PPO(
        "CnnPolicy",
        envir,
        verbose=0,
        learning_rate=model_space.lr,
        max_grad_norm=model_space.max_grad_norm,
        n_steps=model_space.n_steps,
        batch_size=model_space.batch_size,
        n_epochs=model_space.n_epochs,
        device=model_space.dev,
        ent_coef=0.01,
        clip_range=0.1,
        vf_coef=0.5,
    )
    # ----------------- Logger -----------------
    paths_space = PathsParameters(exp_name="ppo", log_dir="breakout_logs")
    logs_config = LogsConfig(paths_space.log_dir,
                             metrics_save_freq=1000,
                             weights_save_freq=20000,
                             videos_save_freq=20000)
    print(paths_space)
    print(logs_config)
    logger = SmartLogger(model.__class__.__name__, options=logs_config, exp_name=paths_space.exp_name)
    # ----------------- Services -----------------
    services: tuple[Support, ...] = (
        EvaluateReward(model, build_env(), freq=logger.options.weights_save_freq, n_episodes=100),
        Checkpointer(model, freq=logger.options.weights_save_freq, path=logger.model_paths[model.__class__.__name__]),
        VideoWriter(model, build_env, freq=logger.options.videos_save_freq, path=os.path.join(logger.path, "videos")),
    )
    # ----------------- Resuming train process -----------------
    last_upd: Optional[str] = logger.get_last_update(model.__class__.__name__)
    model = model.load(last_upd, env=envir, device=model_space.dev) if last_upd else model
    # ----------------- Make some changes -----------------
    model.clip_range = ConstantSchedule(0.2)
    model.ent_coef = 0.02
    # ----------------- Training -----------------
    total_timesteps: int = int(1e4) * model_space.n_parallel * model_space.n_steps
    callback = Callback(*services, writer=logger, show_progress=total_timesteps // model_space.n_parallel)
    model = model.learn(total_timesteps=total_timesteps, callback=callback)
