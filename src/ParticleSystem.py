from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.ForceCalculator import ForceCalculator


@dataclass
class ParticleData:
    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    forces: Optional[np.ndarray] = None

    def __post_init__(self):
        # Data validation
        n = len(self.masses)

        if self.masses.shape != (n,):
            raise ValueError("masses must have shape (n,)")

        if self.positions.shape != (n, 3):
            raise ValueError("position must have shape (n, 3)")

        if self.velocities.shape != (n, 3):
            raise ValueError("velocity must have shape (n, 3)")

        if self.forces is None:
            self.forces = np.zeros(n, 3)
        elif self.forces.shape != (n, 3):
            raise ValueError("force must have shape (n, 3)")

    @property
    def n_particles(self) -> int:
        return len(self.masses)

    def copy(self):
        return ParticleData(
            masses=self.masses.copy(),
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy() if self.forces is not None else None
        )


class ParticleSystem:
    def __init__(self, data: ParticleData, force_calculator: ForceCalculator):
        self.data = data
        self.force_calculator = force_calculator

    def copy(self):
        return ParticleSystem(data=self.data.copy(), force_calculator=self.force_calculator)

    def calc_forces(self):
        forces = self.force_calculator.calc_force(self.data)
        self.data.forces = forces
        return forces

    def calc_keenetic_energy(self):
        keenetic_energy = 0.0
        for i in range(self.data.n_particles):
            keenetic_energy += self.data.masses[i] * self.data.velocities[i]**2 / 2.0
        return keenetic_energy

    def calc_potential_energy(self):
        potential_energy = self.force_calculator.calc_potential_energy(self.data)
        return potential_energy

    def calc_total_energy(self):
        potential_energy = self.calc_potential_energy()
        keenetic_energy = self.calc_keenetic_energy()
        return potential_energy + keenetic_energy

