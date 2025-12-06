import numpy as np

from src import (
    ParticleData,
    Simulation,
    SimulationParameters,
    DirectForceCalculator,
    LFIntegrator,
    Animator2D,
)


def main():
    # Initial data (Sun + Earth)
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    earth_distance = 1.496e11
    earth_velocity = 29780.0

    initial_data = ParticleData(
        masses=np.array([[sun_mass], [earth_mass]]),
        positions=np.array([
            [0.0, 0.0, 0.0],
            [earth_distance, 0.0, 0.0]
        ]),
        velocities=np.array([
            [0.0, 0.0, 0.0],
            [0.0, earth_velocity, 0.0]
        ]),
    )

    print(f"🌍 Number of particles: {initial_data.n_particles}")

    # Simulation parameters
    parameters = SimulationParameters(
        force_calculator=DirectForceCalculator(),
        integrator=LFIntegrator(),
        dt=3600.0  # 1 hour
    )

    simulation = Simulation(initial_data, parameters)

    # Run simulation
    num_steps = 8766  # ~1 year
    positions_history = []

    print("🚀 Running simulation...")

    for i in range(num_steps):
        positions_history.append(simulation.get_data().positions.copy())
        simulation.step()

        if i % 730 == 0:
            progress = 100 * i / num_steps
            print(f"   Progress: {progress:.1f}% ({i}/{num_steps} steps)")

    positions_history = np.array(positions_history)

    print("✅ Simulation complete! Creating animation...")

    # Create animator
    animator = Animator2D(
        positions_history=positions_history,
        masses=initial_data.masses,
        dt=parameters.dt,
        particle_names=['Sun', 'Earth'],
        particle_colors=['yellow', 'dodgerblue'],
        title='Earth Orbit Around Sun',
        show_trails=True,
        dark_theme=True,
    )

    # Show animation
    animator.visualize(interval=1, repeat=True)

    # Optionally save
    # animator.save('earth_orbit.mp4', fps=60, dpi=150)


if __name__ == "__main__":
    main()