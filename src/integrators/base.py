from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation import ParticleSystem


class Integrator(ABC):
    @abstractmethod
    def step(self, system: 'ParticleSystem', dt: float) -> 'ParticleSystem':
        pass
