from abc import ABC, abstractmethod
from src.ParticleSystem import ParticleSystem


class Integrator(ABC):
    @abstractmethod
    def step(self, system: ParticleSystem, dt: float) -> ParticleSystem:
        pass

class ExplicitEulerIntegrator(Integrator):
    def step(self, system: ParticleSystem, dt: float):
        system.data.positions = system.data.positions + system.data.velocities * dt
        system.data.velocities = system.data.velocities + system.data.forces / system.data.masses * dt
        system.calc_forces()
        return system


class RK4Integrator(Integrator):
    def step(self, system: ParticleSystem, dt: float):
        # t0
        pos0 = system.data.positions.copy()
        vel0 = system.data.velocities.copy()

        # k1
        system.calc_forces()
        k1_v = system.data.forces / system.data.masses
        k1_x = system.data.velocities.copy()

        # k2
        system.data.positions = pos0 + k1_x * (dt / 2)
        system.data.velocities = vel0 + k1_v * (dt / 2)
        system.calc_forces()
        k2_v = system.data.forces / system.data.masses
        k2_x = system.data.velocities.copy()

        # k3
        system.data.positions = pos0 + k2_x * (dt / 2)
        system.data.velocities = vel0 + k2_v * (dt / 2)
        system.calc_forces()
        k3_v = system.data.forces / system.data.masses
        k3_x = system.data.velocities.copy()

        # k4
        system.data.positions = pos0 + k3_x * dt
        system.data.velocities = vel0 + k3_v * dt
        system.calc_forces()
        k4_v = system.data.forces / system.data.masses
        k4_x = system.data.velocities.copy()

        # Final update
        system.data.positions = pos0 + dt / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        system.data.velocities = vel0 + dt / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        system.calc_forces()

        return system