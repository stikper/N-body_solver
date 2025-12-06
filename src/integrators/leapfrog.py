from typing import TYPE_CHECKING

from .base import Integrator

if TYPE_CHECKING:
    from src.simulation import ParticleSystem


class LFIntegrator(Integrator):
    def step(self, system: 'ParticleSystem', dt: float):
        # KICK 1: v(t) -> v(t+dt/2)
        system.calc_forces()
        accelerations = system.data.forces / system.data.masses
        system.data.velocities = system.data.velocities + accelerations * (dt / 2)

        # DRIFT: x(t) -> x(t+dt)
        system.data.positions = system.data.positions + system.data.velocities * dt

        # KICK 2: v(t+dt/2) -> v(t+dt)
        system.calc_forces()
        accelerations = system.data.forces / system.data.masses
        system.data.velocities = system.data.velocities + accelerations * (dt / 2)

        return system