from abc import ABC, abstractmethod
from ParticleData import *
from constants import *
import numpy as np


class ForceCalculator(ABC):
    @abstractmethod
    def calc_force(self, data: ParticleData):
        pass

    @abstractmethod
    def calc_potential_energy(self, data: ParticleData):
        pass


class DirectForceCalculator(ForceCalculator):
    def calc_force(self, data: ParticleData) -> np.ndarray:
        forces = np.zeros_like(data.forces)
        for i in range(data.n_particles):
            forces[i] = np.zeros(3)
            for j in range(data.n_particles):
                if i == j:
                    continue
                r = data.positions[j] - data.positions[i]
                forces[i] += (G * data.masses[i] * data.masses[j] / np.linalg.norm(r) ** 3) * r
        return forces

    def calc_potential_energy(self, data: ParticleData) -> float:
        u: float = 0.0
        for i in range(data.n_particles):
            for j in range(data.n_particles):
                if i == j:
                    continue
                l = (data.positions[j] - data.positions[i]).magnitude
                u += -G * data.masses[i] * data.masses[j] / l
        u /= 2
        return u
