from dataclasses import dataclass
from ForceCalculator import ForceCalculator
from Integrator import Integrator

@dataclass
class SimulationParameters:
    force_calculator: ForceCalculator
    integrator: Integrator
    dt: float