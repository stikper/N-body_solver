from abc import ABC, abstractmethod
import numpy as np

from src.core import ParticleData


class ForceCalculator(ABC):
    @abstractmethod
    def calc_force(self, data: ParticleData) -> np.ndarray:
        pass

    @abstractmethod
    def calc_potential_energy(self, data: ParticleData) -> float:
        pass


