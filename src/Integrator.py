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
        explicit_integrator = ExplicitEulerIntegrator()
        k1 = system.copy()
        k2 = explicit_integrator.step(k1.copy(), dt / 2)
        k3 = explicit_integrator.step(k2.copy(), dt / 2)
        k4 = explicit_integrator.step(k3.copy(), dt)

        # return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        system.data.positions = (system.data.positions +
                                 dt / 6 * (k1.data.velocities + 2 * k2.data.velocities +
                                           2 * k3.data.velocities + k4.data.velocities))
        system.data.velocities = (system.data.velocities + dt / 6 * (k1.data.forces / k1.data.masses +
                                                                     2 * k2.data.forces / k2.data.masses +
                                                                     2 * k3.data.forces / k3.data.masses +
                                                                     k4.data.forces / k4.data.masses))
        system.calc_forces()
        return system