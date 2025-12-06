from ForceCalculator import *
from Integrator import *

@dataclass
class SimulationParameters:
    force_calculator: ForceCalculator
    integrator: Integrator
    dt: float

class Simulation:
    def __init__(self, data: ParticleData, sim_params: SimulationParameters):
        self.system = ParticleSystem(data, sim_params.force_calculator)
        self.integrator = sim_params.integrator
        self.dt = sim_params.dt
        self.time = 0.0

    def step(self):
        self.integrator.step(self.system, self.dt)
        self.time += self.dt

    def get_data(self) -> ParticleData:
        return self.system.data

