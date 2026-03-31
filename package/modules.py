import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_, get_total_norm
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule
from torchrl.modules import MLP, ConvNet
from torchrl.objectives import SoftUpdate, DQNLoss
from torchrl.data.tensor_specs import TensorSpec
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.collectors import Collector
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional, Generator
# Package.
from dqn_types import ModelParameters, EnvSpaceName
from utils import except_keyboard_interrupt, fill_buffer
from video import Recorder
from Logger import SmartLogger


class Scale(nn.Module):
    """
    A simple scaling module that divides the input by a constant value.
    Used for normalizing pixel values (e.g., dividing by 255).
    """

    def __init__(self, value: float):
        super().__init__()
        self.value: float = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.value


class PositionalEncoding(nn.Module):
    """
    Implements learnable positional encoding for the Transformer-based DQN.
    Adds a learnable embedding to the input features.
    Supports "Lazy" initialization.
    """

    def __init__(self, shape: Optional[tuple[int, ...]] = None):
        super().__init__()
        self.inited: bool = shape is not None
        if shape is not None:
            self.pos_embedding = nn.Parameter(torch.randn(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inited:
            self.pos_embedding = nn.Parameter(torch.randn_like(x[0]))
        return x + self.pos_embedding


class TransformerDQN(nn.Module):
    """
    A Transformer-based Dueling DQN architecture for visual processing.

    This network uses a Convolutional backbone to compress images, followed by
    a Transformer Encoder for temporal/spatial reasoning over stacked frames,
    and finally Dueling heads for Value and Advantage estimation.
    """

    def __init__(self, n_actions: int, stack_size: int = 4, embed_dim: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead, dim_feedforward=embed_dim * 2, batch_first=True, dropout=0.0
        )
        self.compression = ConvNet(num_cells=(32, 64, 128, 256, 512), kernel_sizes=3, strides=2)
        self.aline = nn.LazyLinear(out_features=embed_dim)
        self.pos_embedding = PositionalEncoding(shape=(1, stack_size, embed_dim))
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.advantage = MLP(out_features=n_actions, depth=2, num_cells=512, activation_class=nn.ReLU)
        self.value = MLP(out_features=1, depth=2, num_cells=512, activation_class=nn.ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_unbatched: bool = x.dim() == 3
        x = x.unsqueeze(0) if is_unbatched else x
        b, s, h, w = x.shape
        x_reshaped = x.view(b * s, 1, h, w)
        features = self.compression(x_reshaped)
        features = features.view(b, s, -1)
        features = self.aline(features)  # (b, stack_size, embed_dim)
        features = self.pos_embedding(features)
        trans_out = self.transformer(features)
        latent = trans_out[:, -1, :]  # (b, embed_dim)
        advantage = self.advantage(latent)
        value = self.value(latent)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values.squeeze(0) if is_unbatched else q_values


class RecurrentDQN(nn.Module):
    def __init__(self, n_actions: int, embed_dim: int = 256):
        super().__init__()
        self.compression = ConvNet(num_cells=(32, 64, 128, 256), kernel_sizes=3, strides=2)
        self.aline = nn.LazyLinear(out_features=embed_dim)
        self.rnn = nn.LSTM(embed_dim, embed_dim, batch_first=True, dropout=0.0)
        self.advantage = MLP(out_features=n_actions, depth=2, num_cells=256, activation_class=nn.ReLU)
        self.value = MLP(out_features=1, depth=2, num_cells=256, activation_class=nn.ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Awaiting input shape: (bs, frames, height, width) or unbatched shape: (frames, height, width)."""
        is_unbatched: bool = x.dim() == 3
        x = x.unsqueeze(0) if is_unbatched else x
        b, s, h, w = x.shape
        x_reshaped = x.view(b * s, 1, h, w)
        features = self.compression(x_reshaped)
        features = features.view(b, s, -1)
        features = self.aline(features)
        _, (hn, _) = self.rnn(features)
        hn = hn[-1]
        advantage = self.advantage(hn)
        value = self.value(hn)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values.squeeze(0) if is_unbatched else q_values


class Model(TensorDictSequential):
    """
    A sequential model wrapper for TensorDictModules.
    Enables grouping multiple modules (e.g., actor and explorer) into a single pipeline.
    """

    def __init__(self, **modules: TensorDictModule):
        super().__init__(OrderedDict([(name, module) for name, module in modules.items()]))


class Optimizer:
    """ Manages the optimization process,
        including loss calculation, gradient clipping,
        and learning rate scheduling. """

    def __init__(
            self,
            network: TensorDictModule,
            action_space: TensorSpec,
            params: ModelParameters,
    ):
        self.loss_fn = DQNLoss(
            network, loss_function="smooth_l1", action_space=action_space, double_dqn=True, delay_value=True
        )
        self.optimizer = torch.optim.Adam(self.loss_fn.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        self.trg_updater = SoftUpdate(self.loss_fn, eps=params.soft_update_eps)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=params.n_epochs, eta_min=params.min_lr)
        self.clipping = params.max_grad_norm
        self.__epoch: int = 0

    def zero_grad(self) -> None:
        """ Resets gradients. """
        self.loss_fn.zero_grad()

    def parameters(self) -> Generator[torch.Tensor]:
        """ Returns the model's parameters. """
        return self.loss_fn.parameters()

    def step(self, loss: torch.Tensor) -> dict[str, float]:
        """ Performs an optimization step with gradient clipping. """
        self.zero_grad()
        loss.backward()
        # -------------------------
        grad_norm: float = clip_grad_norm_(self.loss_fn.parameters(), self.clipping).item()
        weights_norm: float = get_total_norm(self.loss_fn.parameters()).item()
        # -------------------------
        self.optimizer.step()
        self.trg_updater.step()
        self.scheduler.step()
        if hasattr(self.loss_fn.value_network, "explorer"): self.loss_fn.value_network.get_submodule("explorer").step()
        # -------------------------
        last_lr: float = float(self.scheduler.get_last_lr()[0])
        return dict(grad_norm=grad_norm, weights_norm=weights_norm, lr=last_lr)


class Trainer:
    """
    Orchestrates the training process for the DQN agent.
    Handles experience collection, buffer updates, optimization steps, and
    periodic hooks such as logging and checkpointing.
    """

    def __init__(
            self,
            network: TensorDictModule,
            optim: Optimizer,
            params: ModelParameters,
            variables: EnvSpaceName,
            manager: SmartLogger,
            recorder: Recorder
    ):
        self.network: TensorDictModule = network
        self.optim: Optimizer = optim
        self.params: ModelParameters = params
        self.variables: EnvSpaceName = variables
        self.manager: SmartLogger = manager
        self.recorder = recorder
        self._epoch: int = 0

    @staticmethod
    def calc_priorities(td_error: float | torch.Tensor, eps: float = 1e-8) -> float:
        """
        Calculates priority values for the Replay Buffer.
        The priority is computed as the absolute TD error plus a small epsilon
        to ensure all transitions have a non-zero probability of being sampled.

        Args:
            td_error: The Temporal Difference error.
            eps: A small constant for stability.

        Returns:
            The calculated priority.
        """
        return td_error.abs() + eps

    def update_priority(self, rb: ReplayBuffer, idx: torch.Tensor, td_error: float | torch.Tensor) -> None:
        """
        Updates the priorities in the replay buffer based on the TD error.

        Args:
            rb: The Replay Buffer to update.
            idx: The indices of the sampled transitions.
            td_error: The calculated TD error for each transition.
        """
        rb.update_priority(idx, self.calc_priorities(td_error))

    @torch.enable_grad()
    def evaluate(self, rb: ReplayBuffer) -> tuple[TensorDict, TensorDict, TensorDict]:
        """
        Samples data from the replay buffer and calculates the loss.

        Args:
            rb: The Replay Buffer to sample from.

        Returns:
            A tuple containing the sampled batch, info dictionary, and the loss TensorDict.
        """
        sample, info = rb.sample(self.params.batch_size, return_info=True)
        # After computing "loss", sample dict initialize field "td_error".
        sample = sample.to(self.params.dev)
        loss: TensorDict = self.optim.loss_fn(sample)
        return sample, info, loss

    def train_step(self, rb: ReplayBuffer) -> dict[str, int | float]:
        """
        Executes a single training step.
        Includes sampling from the buffer, updating priorities, and performing
        an optimization step.

        Args:
            rb: The Replay Buffer to sample from.

        Returns:
            A dictionary containing step metrics (loss, reward, etc.).
        """
        sample, info, loss = self.evaluate(rb)
        self.update_priority(rb, info[self.variables.index], sample[self.variables.td_error])
        optim_step_info: dict[str, float] = self.optim.step(loss=loss[self.variables.loss])
        step_info: dict[str, int | float] = dict(loss=loss[self.variables.loss].detach().item())
        overlap: set[str] = set(loss.keys()) & set(optim_step_info.keys())
        assert overlap == set(), f"Metrics keys overlap: {overlap}"
        return step_info | optim_step_info

    def perform_hooks(self, data: dict[str, int | float]) -> None:
        """
        Executes periodic background tasks.
        These tasks include:
        - Updating the target network.
        - Stepping the explorer (e.g., epsilon decay).
        - Logging metrics.
        - Saving model checkpoints.
        - Recording gameplay videos.

        Args:
            data: A dictionary containing metrics from the current training step.
        """
        if self._epoch % self.manager.options.metrics_save_freq == 0: self.manager.set_scalars(**data)
        if self._epoch % self.manager.options.weights_save_freq == 0:
            self.manager.checkpoint(self.network.state_dict(), self.variables.actor)
        if self._epoch % self.manager.options.videos_save_freq == 0: self.recorder.shoot(200, self.network)

    @except_keyboard_interrupt
    def train(self, n_epochs: int, rb: ReplayBuffer, loader: Collector, show: bool = True) -> None:
        """
        Main training loop.

        Args:
            n_epochs: Total number of training iterations.
            rb: The Replay Buffer to sample from.
            loader: The Collector to gather new experiences.
            show: Whether to display a progress bar.
        """
        self.network.to(self.params.dev)
        progress_bar = tqdm(range(n_epochs)) if show else range(n_epochs)
        # Filling buffer by new experience, sampling and optimize "actor", make a background process, repeat ...
        for _ in progress_bar:
            self._epoch += 1
            reward: float = fill_buffer(loader, rb, self.params.rb_expansion, show=False)
            step_info: dict[str, int | float] = self.train_step(rb)
            step_info[self.variables.reward] = reward
            self.perform_hooks(step_info)
            # Update progress bar description.
            desc: str = "; ".join([f"{key.capitalize()}: {value:.5f}" for key, value in step_info.items()])
            progress_bar.set_description(desc)


def initialize_weights(module: nn.Module) -> None:
    """
    Initializes weights for different types of neural network modules.

    Uses Kaiming initialization for Convolutional layers and Xavier initialization for Linear
    and Transformer layers.

    Args:
        module: The neural network module to initialize.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)


@torch.no_grad()
def init_lazy_layers(tensordict: TensorDict, network: TensorDictModule) -> TensorDictModule:
    """
    Performs a dummy forward pass to initialize lazy layers in the network.

    Args:
        tensordict: A sample input TensorDict.
        network: The network with lazy layers to initialize.

    Returns:
        The initialized network.
    """
    _ = network(tensordict)
    return network


def n_parameters(module: nn.Module) -> int:
    """
    Calculates the total number of trainable parameters in a module.

    Args:
        module: The neural network module.

    Returns:
        The total count of parameters.
    """
    total: int = 0
    for param in module.parameters():
        total += np.prod(list(param.shape))
    return total
