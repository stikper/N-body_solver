from dataclasses import dataclass
from src.physics import ForceCalculator
from src.integrators import Integrator

@dataclass
class SimulationParameters:
    force_calculator: ForceCalculator
    integrator: Integrator
    dt: float