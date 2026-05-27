import torch
import numpy as np

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.step = 0

    @abstractmethod
    def act(self, step) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def update(self, action: list[float], reward: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_using_batch_gd(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_to_disk(self, path) -> None:
        raise NotImplementedError
