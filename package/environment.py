from gymnasium import Env, Wrapper, make
from torchrl.envs import GymWrapper
from typing import Callable, Iterator, Any
from collections import OrderedDict


class RewardOnLifeLoss(Wrapper):
    """ Wrapper for Atari games that penalizes the agent for losing a life. """

    def __init__(self, env: Env, penalty_weight: float = 1.0):
        super().__init__(env)
        self.lives: int = 0
        self.penalty_weight = penalty_weight

    def step(self, action: int) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get("lives", 0)
        if (current_lives < self.lives) and (self.lives > 0): reward -= self.penalty_weight
        self.lives = current_lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> Any:
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        return obs, info


def env2torch(func: Callable[..., Env]):
    """ Decorator. Transfer "Env" from gymnasium into torchrl "GymWrapper". """

    def wrapper(*args, **kwargs) -> GymWrapper:
        env: Env = func(*args, **kwargs)
        torch_env: GymWrapper = GymWrapper(env, from_pixels=True)
        return torch_env

    return wrapper


class GymPreprocessing(OrderedDict):
    def __init__(self, *wrappers: Callable[[Env], Env]):
        indexes: range = range(len(wrappers))
        pairs: Iterator[tuple[int, Env]] = zip(indexes, wrappers)
        super().__init__(pairs)

    def forward(self, env: Env) -> Env:
        result: Env = env
        for idx, wrap in self.items(): result = wrap(result)
        return result

    def __call__(self, env: Env) -> Env:
        return self.forward(env)


def create_breakout_env_gym(name: str = "ALE/Breakout-v5", transform: Callable[[Env], Env] = None) -> Env:
    environment = make(name, render_mode="rgb_array", frameskip=1)
    environment = transform(environment) if (transform is not None) else environment
    return environment


@env2torch
def create_breakout_env(name: str = "ALE/Breakout-v5", transform: Callable[[Env], Env] = None) -> GymWrapper:
    return create_breakout_env_gym(name, transform)
