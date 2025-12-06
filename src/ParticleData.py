from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ParticleData:
    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    forces: Optional[np.ndarray] = None

    def __post_init__(self):
        # Data validation
        n = len(self.masses)

        if self.masses.shape != (n, 1):
            raise ValueError("masses must have shape (n, 1)")

        if self.positions.shape != (n, 3):
            raise ValueError("position must have shape (n, 3)")

        if self.velocities.shape != (n, 3):
            raise ValueError("velocity must have shape (n, 3)")

        if self.forces is None:
            self.forces = np.zeros((n,3))
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


