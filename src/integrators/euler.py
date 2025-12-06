from typing import TYPE_CHECKING

from .base import Integrator

if TYPE_CHECKING:
    from src.simulation import ParticleSystem

class ExplicitEulerIntegrator(Integrator):
    def step(self, system: 'ParticleSystem', dt: float):
        system.data.positions = system.data.positions + system.data.velocities * dt
        system.data.velocities = system.data.velocities + system.data.forces / system.data.masses * dt
        system.calc_forces()
        return system