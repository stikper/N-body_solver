from src.core import ParticleData
from src.physics import ForceCalculator


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

