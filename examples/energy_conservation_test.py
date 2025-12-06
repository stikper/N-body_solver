"""
Тест сохранения энергии для разных интеграторов.
"""

import numpy as np
import matplotlib.pyplot as plt

from src import (
    ParticleData,
    Simulation,
    SimulationParameters,
    DirectForceCalculator,
    ExplicitEulerIntegrator,
    LFIntegrator,
    RK4Integrator,
    EnergyPlotter,
)


def create_earth_sun_system():
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    earth_distance = 1.496e11
    earth_velocity = 29780.0

    return ParticleData(
        masses=np.array([[sun_mass], [earth_mass]]),
        positions=np.array([[0.0, 0.0, 0.0], [earth_distance, 0.0, 0.0]]),
        velocities=np.array([[0.0, 0.0, 0.0], [0.0, earth_velocity, 0.0]]),
    )


def run_simulation(integrator, dt, num_steps):
    data = create_earth_sun_system()
    params = SimulationParameters(
        force_calculator=DirectForceCalculator(),
        integrator=integrator,
        dt=dt
    )
    sim = Simulation(data, params)

    times = []
    kinetic = []
    potential = []
    total = []

    for i in range(num_steps):
        times.append(sim.time)
        kinetic.append(sim.system.calc_kinetic_energy())
        potential.append(sim.system.calc_potential_energy())
        total.append(sim.system.calc_total_energy())
        sim.step()

    return (
        np.array(times),
        np.array(kinetic),
        np.array(potential),
        np.array(total)
    )


def main():
    print("🧪 Testing energy conservation for different integrators...")

    dt = 36000.0  # 1 hour
    num_steps = 8766  # ~1 year

    # Test Euler
    print("   🐌 Euler integrator...")
    times_euler, ke_euler, pe_euler, te_euler = run_simulation(
        ExplicitEulerIntegrator(), dt, num_steps
    )

    # Test Leap-Frog
    print("   🐸 Leap-Frog integrator...")
    times_lf, ke_lf, pe_lf, te_lf = run_simulation(
        LFIntegrator(), dt, num_steps
    )

    # Test RK4
    print("   🚀 RK4 integrator...")
    times_rk4, ke_rk4, pe_rk4, te_rk4 = run_simulation(
        RK4Integrator(), dt, num_steps
    )

    print("✅ All simulations complete!")

    # Calculate relative errors
    E0_euler = te_euler[0]
    E0_lf = te_lf[0]
    E0_rk4 = te_rk4[0]

    rel_err_euler = np.abs((te_euler - E0_euler) / E0_euler) * 100
    rel_err_lf = np.abs((te_lf - E0_lf) / E0_lf) * 100
    rel_err_rk4 = np.abs((te_rk4 - E0_rk4) / E0_rk4) * 100

    print("\n📊 Energy conservation statistics:")
    print(f"   Euler:     max error = {rel_err_euler.max():.6f}%")
    print(f"   Leap-Frog: max error = {rel_err_lf.max():.6f}%")
    print(f"   RK4:       max error = {rel_err_rk4.max():.6f}%")

    # Plot comparison
    print("\n🎨 Creating comparison plots...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Dark theme
    fig.patch.set_facecolor('#001d3d')
    for ax in axes:
        ax.set_facecolor('#000814')
        ax.grid(True, alpha=0.2, color='cyan')
        ax.tick_params(colors='white')

    times_days_euler = times_euler / 86400.0
    times_days_lf = times_lf / 86400.0
    times_days_rk4 = times_rk4 / 86400.0

    # Plot 1: Total energy
    ax1 = axes[0]
    ax1.plot(times_days_euler, te_euler, 'r-', label='Euler', linewidth=1.5, alpha=0.8)
    ax1.plot(times_days_lf, te_lf, 'g-', label='Leap-Frog', linewidth=1.5, alpha=0.8)
    ax1.plot(times_days_rk4, te_rk4, 'b-', label='RK4', linewidth=1.5, alpha=0.8)

    ax1.set_ylabel('Total Energy [J]', color='white', fontsize=12)
    ax1.set_title('⚡ Total Energy Comparison', color='yellow', fontsize=16, weight='bold')
    ax1.legend(loc='best', fontsize=10, facecolor='black', edgecolor='cyan', labelcolor='white')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Plot 2: Relative error
    ax2 = axes[1]
    ax2.plot(times_days_euler, rel_err_euler, 'r-', label='Euler', linewidth=1.5, alpha=0.8)
    ax2.plot(times_days_lf, rel_err_lf, 'g-', label='Leap-Frog', linewidth=1.5, alpha=0.8)
    ax2.plot(times_days_rk4, rel_err_rk4, 'b-', label='RK4', linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('Time [days]', color='white', fontsize=12)
    ax2.set_ylabel('Relative Error [%]', color='white', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=10, facecolor='black', edgecolor='cyan', labelcolor='white')

    plt.tight_layout()

    # Save figure
    filename = 'energy_comparison.png'
    print(f"💾 Saving comparison plot to '{filename}'...")
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#001d3d')
    print(f"✅ Saved to '{filename}'!")

    plt.show()

    # Individual energy plots using EnergyPlotter
    print("\n🎨 Creating individual energy plots...")

    # Leap-Frog energy plot
    plotter_lf = EnergyPlotter(
        times=times_lf,
        kinetic_energies=ke_lf,
        potential_energies=pe_lf,
        total_energies=te_lf,
        title='⚡ Energy Conservation - Leap-Frog',
        dark_theme=True
    )
    plotter_lf.visualize(show_relative_error=True, show=False)
    # plotter_lf.save('energy_leapfrog.png', dpi=300)

    # RK4 energy plot
    plotter_rk4 = EnergyPlotter(
        times=times_rk4,
        kinetic_energies=ke_rk4,
        potential_energies=pe_rk4,
        total_energies=te_rk4,
        title='⚡ Energy Conservation - RK4',
        dark_theme=True
    )
    plotter_rk4.visualize(show_relative_error=True, show=False)
    # plotter_rk4.save('energy_rk4.png', dpi=300)

    # Euler energy plot
    plotter_euler = EnergyPlotter(
        times=times_euler,
        kinetic_energies=ke_euler,
        potential_energies=pe_euler,
        total_energies=te_euler,
        title='⚡ Energy Conservation - Euler',
        dark_theme=True
    )
    plotter_euler.visualize(show_relative_error=True, show=True)
    # plotter_euler.save('energy_euler.png', dpi=300)

    print("\n🎉 All done! Check the saved plots!")


if __name__ == "__main__":
    main()