from abc import ABC, abstractmethod
import numpy as np


class Visualizer(ABC):
    @abstractmethod
    def visualize(self, data: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def save(self, filename: str, **kwargs):
        pass