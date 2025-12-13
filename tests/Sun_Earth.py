import numpy as np
import argparse
from src import (
    ParticleData, G, AU,
    DirectForceCalculator,
    LFIntegrator, RK4Integrator, ExplicitEulerIntegrator,
    Simulation, SimulationParameters,
    Animator2D, Plotter2D, EnergyPlotter
)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='🌍 Симуляция движения Земли вокруг Солнца на 1 год',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --method lf --dt 3600
  python main.py --method rk4 --dt 1800
  python main.py --method euler --dt 900 --no-animation
        """
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['euler', 'rk4', 'lf'],
        default='lf',
        help='Метод интегрирования (euler/rk4/lf). По умолчанию: lf'
    )

    parser.add_argument(
        '--dt',
        type=float,
        default=36000.0,
        help='Временной шаг в секундах. По умолчанию: 3600 (1 час)'
    )

    parser.add_argument(
        '--no-animation',
        action='store_true',
        help='Не показывать анимацию, только графики'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Сохранить графики и анимацию'
    )

    return parser.parse_args()


def get_integrator(method_name):
    """Получить интегратор по имени"""
    integrators = {
        'euler': ExplicitEulerIntegrator,
        'rk4': RK4Integrator,
        'lf': LFIntegrator,
    }
    return integrators[method_name]()


def main():
    args = parse_args()

    print("=" * 60)
    print("🌍 СИМУЛЯЦИЯ ДВИЖЕНИЯ ЗЕМЛИ ВОКРУГ СОЛНЦА")
    print("=" * 60)
    print(f"📊 Метод интегрирования: {args.method.upper()}")
    print(f"⏱️  Временной шаг: {args.dt} с ({args.dt / 3600:.2f} часов)")
    print("=" * 60)

    # ========== 1. НАЧАЛЬНЫЕ ДАННЫЕ ==========
    # Данные из EPM2021 на 2025-12-13 04:42:28 UTC

    # Массы (в кг) - shape: (n, 1)
    masses = np.array([
        [1.989e30],  # Солнце
        [5.972e24],  # Земля
    ])

    # Позиции (x, y, z в метрах) - shape: (n, 3)
    # Относительно барицентра Солнечной системы
    positions = np.array([
        [-479120095.480, -767705664.734, -310948056.613],  # Солнце
        [22407886361.781, 132721917099.621, 57554472840.542],  # Земля
    ])

    # Скорости (vx, vy, vz в м/с) - shape: (n, 3)
    velocities = np.array([
        [12.513, 0.097, -0.214],  # Солнце
        [-29899.062, 4154.016, 1801.674],  # Земля
    ])

    # Создаём объект данных
    data = ParticleData(
        masses=masses,
        positions=positions,
        velocities=velocities
    )

    print(f"✅ Создано частиц: {data.n_particles}")
    print(f"📍 Начальное расстояние Земля-Солнце: {np.linalg.norm(positions[1] - positions[0]) / AU:.4f} а.е.")

    # ========== 2. ПАРАМЕТРЫ СИМУЛЯЦИИ ==========
    force_calculator = DirectForceCalculator()
    integrator = get_integrator(args.method)

    sim_params = SimulationParameters(
        force_calculator=force_calculator,
        integrator=integrator,
        dt=args.dt
    )

    # ========== 3. ЗАПУСК СИМУЛЯЦИИ ==========
    sim = Simulation(data=data, sim_params=sim_params)

    # Симуляция на 1 год
    year_seconds = 365.25 * 86400.0
    n_steps = int(year_seconds / args.dt)

    print(f"🔄 Количество шагов: {n_steps}")
    print(f"⏳ Симулируемое время: 1 год ({year_seconds / 86400:.1f} дней)")
    print("=" * 60)

    # Массивы для истории
    positions_history = np.zeros((n_steps, data.n_particles, 3))
    velocities_history = np.zeros((n_steps, data.n_particles, 3))
    times = np.zeros(n_steps)
    kinetic_energies = np.zeros(n_steps)
    potential_energies = np.zeros(n_steps)
    total_energies = np.zeros(n_steps)

    # Главный цикл симуляции
    print("🚀 Запуск симуляции...")
    for i in range(n_steps):
        # Сохраняем состояние
        positions_history[i] = sim.get_data().positions
        velocities_history[i] = sim.get_data().velocities
        times[i] = sim.time

        # Энергии
        kinetic_energies[i] = sim.system.calc_kinetic_energy()
        potential_energies[i] = sim.system.calc_potential_energy()
        total_energies[i] = sim.system.calc_total_energy()

        # Шаг
        sim.step()

        # Прогресс
        if i % (n_steps // 20) == 0:
            progress = 100 * i / n_steps
            print(f"  ⏳ Прогресс: {progress:.1f}% ({i}/{n_steps} шагов)")

    print("✅ Симуляция завершена!")
    print("=" * 60)

    # Статистика
    final_distance = np.linalg.norm(
        positions_history[-1, 1] - positions_history[-1, 0]
    ) / AU
    initial_distance = np.linalg.norm(
        positions_history[0, 1] - positions_history[0, 0]
    ) / AU

    energy_error = abs(total_energies[-1] - total_energies[0]) / abs(total_energies[0]) * 100

    print("📊 РЕЗУЛЬТАТЫ:")
    print(f"  🌍 Начальное расстояние: {initial_distance:.6f} а.е.")
    print(f"  🌍 Конечное расстояние: {final_distance:.6f} а.е.")
    print(f"  📏 Изменение расстояния: {abs(final_distance - initial_distance):.6f} а.е.")
    print(f"  ⚡ Ошибка сохранения энергии: {energy_error:.6e}%")
    print("=" * 60)

    # ========== 4. ВИЗУАЛИЗАЦИЯ ==========
    particle_names = ['Sun', 'Earth']
    particle_colors = ['#FDB813', '#1E90FF']  # Золотой и голубой

    # График траекторий
    print("📈 Построение графика траекторий...")
    plotter = Plotter2D(
        positions_history=positions_history,
        masses=masses,
        dt=args.dt,
        particle_names=particle_names,
        particle_colors=particle_colors,
        title=f'Орбита Земли вокруг Солнца (метод: {args.method.upper()}, dt={args.dt}s)',
        dark_theme=True,
    )

    if args.save:
        plotter.save(f'trajectory_{args.method}_dt{int(args.dt)}.png', dpi=300)
        print(f"  ✅ Сохранено: trajectory_{args.method}_dt{int(args.dt)}.png")

    plotter.visualize(show_start=True, show_end=True, show=True)

    # График энергии
    print("⚡ Построение графика энергии...")
    energy_plotter = EnergyPlotter(
        times=times / 86400.0,  # Переводим в дни
        kinetic_energies=kinetic_energies,
        potential_energies=potential_energies,
        total_energies=total_energies,
        title=f'⚡ Сохранение энергии (метод: {args.method.upper()}, dt={args.dt}s)',
        dark_theme=True,
        time_label='Time (days)'
    )

    if args.save:
        energy_plotter.save(f'energy_{args.method}_dt{int(args.dt)}.png', dpi=300)
        print(f"  ✅ Сохранено: energy_{args.method}_dt{int(args.dt)}.png")

    energy_plotter.visualize(show_relative_error=True, show=True)

    # Анимация
    if not args.no_animation:
        print("🎬 Создание анимации...")
        animator = Animator2D(
            positions_history=positions_history,
            masses=masses,
            dt=args.dt,
            particle_names=particle_names,
            particle_colors=particle_colors,
            title=f'🌌 Земля вокруг Солнца (метод: {args.method.upper()})',
            show_trails=True,
            trail_length=1000,
            dark_theme=True,
        )

        if args.save:
            print("  💾 Сохранение анимации (требует ffmpeg)...")
            animator.save(f'animation_{args.method}_dt{int(args.dt)}.mp4', fps=30, dpi=150)
            print(f"  ✅ Сохранено: animation_{args.method}_dt{int(args.dt)}.mp4")

        animator.visualize(interval=10, repeat=True, show=True)

    print("=" * 60)
    print("✨ Готово!")
    print("=" * 60)


if __name__ == '__main__':
    main()