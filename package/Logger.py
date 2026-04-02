import os
import torch
import time
import csv
import pandas as pd
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Any
import matplotlib.pyplot as plt
from utils import get_last_update, read_json, write_json


class Logger:
    """
    Base Logger class for managing experiment logs and checkpoints.
    """

    def __init__(self, directory: str, log_name: str = "logs", checkpoints_dir: str = "checkpoints"):
        """
        Initializes the Logger.

        Args:
            directory: The directory where logs and checkpoints will be stored.
            log_name: The name of the JSON file to store logs.
            checkpoints_dir: The name of the directory to store model checkpoints.
        """
        self.logs_var = "logs"
        self.hyperparams_var = "hyperparams"
        self.directory = directory
        self.logs_file_path = os.path.join(directory, f"{log_name}.json")
        self.checkpoints_dir = os.path.join(directory, checkpoints_dir)
        # Post initialization.
        self.setup()

    def setup(self) -> None:
        """
        Initializes the necessary directories and files if they do not exist.
        """
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        init_logs = {self.hyperparams_var: None, self.logs_var: []}
        if not os.path.exists(self.logs_file_path): write_json(self.logs_file_path, init_logs)

    def checkpoint(self, state_dict: OrderedDict) -> None:
        """
        Saves the current state of a neural network module to a checkpoint file.

        Args:
            state_dict: The state dictionary of the model.
        """
        assert isinstance(state_dict, OrderedDict), f"Checkpoint is not an OrderedDict."
        torch.save(state_dict, os.path.join(self.checkpoints_dir, f"checkpoint_{time.time()}.pt"))

    def log(self, **kwargs: Any) -> None:
        """
        Appends a new entry to the logs file.

        Args:
            **kwargs: Key-value pairs of metrics to log.
        """
        data: dict = read_json(self.logs_file_path)
        assert isinstance(data, dict), f"Logs file {self.logs_file_path} is corrupted."
        assert self.logs_var in data, f"KeyError: '{self.logs_var}' key missing."
        data[self.logs_var].append(kwargs)
        write_json(self.logs_file_path, data)

    def set_hyperparams(self, kwargs: Any) -> None:
        """
        Overwrites the hyperparameters dictionary in the logs file.

        Args:
            kwargs: A dictionary of hyperparameters.
        """
        data: dict = read_json(self.logs_file_path)
        assert isinstance(data, dict), f"Logs file {self.logs_file_path} is corrupted."
        assert self.hyperparams_var in data, f"KeyError: '{self.hyperparams_var}' key missing."
        data[self.hyperparams_var] = kwargs
        write_json(self.logs_file_path, data)

    def get_logs(self) -> dict[str, Any]:
        """ Returns the entire "logs" dictionary. """
        data: dict = read_json(self.logs_file_path)
        return data

    def get_last_checkpoint(self) -> Optional[OrderedDict]:
        """
        Retrieves the most recent checkpoint from the checkpoints directory.

        Returns:
            The state dictionary of the last checkpoint, or None if no checkpoint exists.
        """
        last: Optional[str] = get_last_update(self.checkpoints_dir)
        if last is None: return None
        state = torch.load(last, weights_only=True)
        assert isinstance(state, OrderedDict), f"Not correctly saved checkpoint: {last}, dtype: {type(state)}."
        print(f"Last checkpoint: {last}")
        return state


@dataclass
class LogsConfig:
    """
    Configuration for logging metrics and saving checkpoints.

    Attributes:
        log_dir: Root directory for logs. Default: "logs".
        metrics_save_freq: Frequency of saving scalar metrics (in epochs). Default: 1.
        weights_save_freq: Frequency of saving model checkpoints (in epochs). Default: 20.
        videos_save_freq: Frequency of saving video experience (in epochs). Default: 20.
    """
    log_dir: str = "logs"
    metrics_save_freq: int = 1
    weights_save_freq: int = 20
    videos_save_freq: int = 20


class SmartLogger:
    """ A utility class for logging model weights and scalar metrics during training. """

    def __init__(self, *model_names: str, options: LogsConfig, exp_name: str):
        """ Initialize the Logger with a directory path and experiment name. """
        self.options = options
        self.path = os.path.join(options.log_dir, exp_name)
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.model_paths: dict[str, str] = {name: os.path.join(self.checkpoint_path, name) for name in model_names}
        self.scalars_path = os.path.join(self.path, "scalars")
        self._weights_ext = "pt"
        self.init()

    def init(self) -> None:
        """ Create the necessary directories for checkpoints and scalars. """
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.scalars_path, exist_ok=True)
        for model in self.model_paths.keys():
            os.makedirs(os.path.join(self.checkpoint_path, model), exist_ok=True)

    def checkpoint(self, weights: dict, model: str) -> None:
        """ Save model weights as a checkpoint file. """
        assert model in self.model_paths.keys(), f"Logger does not know model: {model}."
        torch.save(weights, os.path.join(self.model_paths[model], f"weights_{time.time()}.{self._weights_ext}"))

    def set_scalars(self, **scalars: int | float) -> None:
        """ Log multiple scalar values with the current timestamp. """
        for name, value in scalars.items():
            self.set_scalar(time.time(), name, value)

    def set_scalar(self, dtime: float, name: str, value: int | float | str) -> None:
        """ Log a single scalar value to a CSV file. """
        file_path: str = os.path.join(self.scalars_path, f"{name}.csv")
        file_exists: bool = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(["dtime", "value"])
            writer.writerow([dtime, value])

    def get_last_update(self, model: str) -> Optional[str]:
        """ Retrieve the path to the most recent checkpoint file in the directory. """
        assert model in self.model_paths.keys(), f"Logger does not know model: {model}."
        return get_last_update(self.model_paths[model])

    def draw_scalars(self, exclude: Optional[list[str]] = None) -> None:
        """
        Plots logged metrics from CSV files.
        Automatically creates subplots for each metric found in the scalar's directory.
        """
        assert os.path.exists(self.scalars_path), f"Directory {self.scalars_path} does not exist."
        csv_files: list[str] = [file for file in os.listdir(self.scalars_path) if file.endswith(".csv")]
        if exclude is not None:
            # TODO: Implement exclude argument.
            raise NotImplementedError("Not implemented exclude argument yet.")
        if len(csv_files) == 0:
            print("No CSV files found.")
        else:
            n_files: int = len(csv_files)
            fig, axes = plt.subplots(n_files, 1, figsize=(5, 4 * n_files), sharex=False)
            axes = [axes] if (n_files == 1) else axes
            for idx, file in enumerate(csv_files):
                file_path: str = os.path.join(self.scalars_path, file)
                df: pd.DataFrame = pd.read_csv(file_path)
                metric_name: str = file.replace(".csv", "")
                # Drawing.
                axes[idx].plot(df["dtime"].index, df["value"], label=metric_name, color="dodgerblue")
                axes[idx].set_title(f"Metric: {metric_name}")
                axes[idx].set_xlabel("Epoch")
                axes[idx].set_ylabel("Value")
                axes[idx].legend()
                axes[idx].grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
