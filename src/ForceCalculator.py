from abc import ABC, abstractmethod
from ParticleSystem import ParticleData
from constants import *
import numpy as np


class ForceCalculator(ABC):
    @abstractmethod
    def compute_force(self, data: ParticleData):
        pass

    @abstractmethod
    def compute_potential_energy(self, data: ParticleData):
        pass


class DirectForceCalculator(ForceCalculator):
    def compute_force(self, data: ParticleData) -> np.ndarray:
        forces = np.zeros_like(data.forces)
        for i in range(data.n_particles):
            forces[i] = np.zeros(3)
            for j in range(data.n_particles):
                if i == j:
                    continue
                r = data.positions[j] - data.positions[i]
                forces[i] += (G * data.masses[i] * data.masses[j] / r.magnitude ** 3) * r
        return forces

    def compute_potential_energy(self, data: ParticleData) -> float:
        u: float = 0.0
        for i in range(data.n_particles):
            for j in range(data.n_particles):
                if i == j:
                    continue
                l = (data.positions[j] - data.positions[i]).magnitude
                u += -G * data.masses[i] * data.masses[j] / l
        u /= 2
        return u
