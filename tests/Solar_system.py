import numpy as np
import argparse
from src import (
    ParticleData, G, AU,
    DirectForceCalculator,
    LFIntegrator, RK4Integrator, ExplicitEulerIntegrator,
    Simulation, SimulationParameters,
    Animator2D, Plotter2D, EnergyPlotter
)


def create_solar_system_data():
    """
    Создаёт ParticleData для реальной Солнечной системы.
    Данные получены из EPM2021 на 2025-12-13 04:42:28 UTC
    относительно барицентра Солнечной системы.
    """

    # Массы небесных тел (в кг)
    # Источник: IAU/NASA стандартные значения
    masses = np.array([
        [1.98841e30],  # Sun (Солнце)
        [3.30110e23],  # Mercury (Меркурий)
        [4.86732e24],  # Venus (Венера)
        [5.97217e24],  # Earth (Земля)
        [6.41693e23],  # Mars (Марс)
        [1.89813e27],  # Jupiter (Юпитер)
        [5.68319e26],  # Saturn (Сатурн)
        [8.68103e25],  # Uranus (Уран)
        [1.02410e26],  # Neptune (Нептун)
        [1.30900e22],  # Pluto (Плутон)
    ])

    # Позиции (X, Y, Z в метрах)
    # Координаты: Equatorial Mean J2000
    positions = np.array([
        [-479120095.480, -767705664.734, -310948056.613],  # Sun
        [-58637939612.900, -4677510733.422, 3628042236.444],  # Mercury
        [-42523165514.879, -92973444841.244, -39140953872.635],  # Venus
        [22407886361.781, 132721917099.621, 57554472840.542],  # Earth
        [10087995256.084, -197521743479.253, -90842337693.048],  # Mars
        [-233468076149.888, 680064632688.194, 297183625263.336],  # Jupiter
        [1423015591043.480, 42942502046.692, -43554480707.443],  # Saturn
        [1487224413928.140, 2304369453427.526, 988215836744.449],  # Uranus
        [4468555127925.526, 103836004405.415, -68749496835.001],  # Neptune
        [2868338108400.359, -3932184130808.958, -2091335882359.679],  # Pluto
    ])

    # Скорости (Vx, Vy, Vz в м/с)
    velocities = np.array([
        [12.513, 0.097, -0.214],  # Sun
        [-8431.777, -41541.575, -21316.719],  # Mercury
        [32058.118, -11778.176, -7327.505],  # Venus
        [-29899.062, 4154.016, 1801.674],  # Earth
        [25129.723, 3213.343, 796.159],  # Mars
        [-12617.943, -3146.411, -1041.450],  # Jupiter
        [-683.173, 8898.142, 3704.785],  # Saturn
        [-5907.321, 2861.175, 1336.658],  # Uranus
        [-118.393, 5059.455, 2073.811],  # Neptune
        [4716.140, 2213.847, -730.083],  # Pluto
    ])

    # Создаём объект ParticleData
    data = ParticleData(
        masses=masses,
        positions=positions,
        velocities=velocities
    )

    return data


def get_planet_names():
    """Возвращает список названий планет для визуализации"""
    return [
        'Sun', 'Mercury', 'Venus', 'Earth', 'Mars',
        'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'
    ]


def get_planet_colors():
    """Возвращает список цветов для визуализации планет"""
    return [
        '#FDB813',  # Sun - золотой
        '#8C7853',  # Mercury - серый
        '#FFC649',  # Venus - оранжевый
        '#1E90FF',  # Earth - голубой
        '#CD5C5C',  # Mars - красный
        '#DAA520',  # Jupiter - оранжевый
        '#F4A460',  # Saturn - песочный
        '#4FD0E0',  # Uranus - светло-голубой
        '#4169E1',  # Neptune - синий
        '#8B7355',  # Pluto - коричневый
    ]


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='🌌 Симуляция Солнечной системы (10 тел)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Симуляция на 1 год с шагом 1 час
  python main.py --years 1 --dt 3600 --method lf

  # Симуляция на 10 лет с шагом 6 часов
  python main.py --years 10 --dt 21600 --method lf

  # Симуляция на 100 лет (только графики, без анимации)
  python main.py --years 100 --dt 86400 --method lf --no-animation

  # Симуляция внутренних планет на 1 год
  python main.py --years 1 --dt 3600 --inner-only

  # Высокоточная симуляция на 1 год
  python main.py --years 1 --dt 1800 --method rk4
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
        default=3600.0,
        help='Временной шаг в секундах. По умолчанию: 3600 (1 час)'
    )

    parser.add_argument(
        '--years',
        type=float,
        default=1.0,
        help='Длительность симуляции в годах. По умолчанию: 1'
    )

    parser.add_argument(
        '--inner-only',
        action='store_true',
        help='Симулировать только внутренние планеты (Меркурий-Марс)'
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

    parser.add_argument(
        '--trail-length',
        type=int,
        default=2000,
        help='Длина следа орбит в анимации. По умолчанию: 2000'
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


def print_orbital_info(data, names):
    """Вывод информации об орбитах"""
    sun_pos = data.positions[0]

    print("\n📊 ОРБИТАЛЬНЫЕ ПАРАМЕТРЫ:")
    print("-" * 80)
    print(f"{'Планета':<12} {'Расстояние (а.е.)':<20} {'Скорость (км/с)':<20}")
    print("-" * 80)

    for i in range(1, data.n_particles):
        distance = np.linalg.norm(data.positions[i] - sun_pos) / AU
        velocity = np.linalg.norm(data.velocities[i]) / 1000.0
        print(f"{names[i]:<12} {distance:>18.4f}   {velocity:>18.2f}")

    print("-" * 80)


def main():
    args = parse_args()

    print("=" * 80)
    print("🌌 СИМУЛЯЦИЯ СОЛНЕЧНОЙ СИСТЕМЫ")
    print("=" * 80)
    print(f"📊 Метод интегрирования: {args.method.upper()}")
    print(f"⏱️  Временной шаг: {args.dt} с ({args.dt / 3600:.2f} часов)")
    print(f"📅 Длительность: {args.years} лет")
    print("=" * 80)

    # ========== 1. СОЗДАНИЕ ДАННЫХ ==========
    full_data = create_solar_system_data()
    planet_names = get_planet_names()
    planet_colors = get_planet_colors()

    # Фильтрация для внутренних планет
    if args.inner_only:
        # Индексы: Sun (0), Mercury (1), Venus (2), Earth (3), Mars (4)
        indices = [0, 1, 2, 3, 4]
        data = ParticleData(
            masses=full_data.masses[indices],
            positions=full_data.positions[indices],
            velocities=full_data.velocities[indices]
        )
        planet_names = [planet_names[i] for i in indices]
        planet_colors = [planet_colors[i] for i in indices]
        print("🔍 Режим: Только внутренние планеты (Меркурий-Марс)")
    else:
        data = full_data
        print("🔍 Режим: Вся Солнечная система (включая Плутон)")

    print(f"✅ Создано объектов: {data.n_particles}")

    # Вывод орбитальной информации
    print_orbital_info(data, planet_names)

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

    # Расчёт количества шагов
    year_seconds = 365.25 * 86400.0
    total_seconds = args.years * year_seconds
    n_steps = int(total_seconds / args.dt)

    print(f"\n🔄 Количество шагов: {n_steps:,}")
    print(f"⏳ Симулируемое время: {args.years} лет ({total_seconds / 86400:.1f} дней)")
    print("=" * 80)

    # Массивы для истории
    positions_history = np.zeros((n_steps, data.n_particles, 3))
    velocities_history = np.zeros((n_steps, data.n_particles, 3))
    times = np.zeros(n_steps)
    kinetic_energies = np.zeros(n_steps)
    potential_energies = np.zeros(n_steps)
    total_energies = np.zeros(n_steps)

    # Главный цикл симуляции
    print("🚀 Запуск симуляции...\n")
    progress_points = 20

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
        if i % (n_steps // progress_points) == 0 or i == n_steps - 1:
            progress = 100 * i / n_steps
            years_done = sim.time / year_seconds
            print(f"  ⏳ Прогресс: {progress:5.1f}% | "
                  f"Шаг: {i:>8,}/{n_steps:,} | "
                  f"Время: {years_done:6.2f} лет")

    print("\n✅ Симуляция завершена!")
    print("=" * 80)

    # ========== СТАТИСТИКА ==========
    print("\n📊 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
    print("-" * 80)

    # Проверка орбит
    sun_pos_initial = positions_history[0, 0]
    sun_pos_final = positions_history[-1, 0]

    print(f"{'Планета':<12} {'Начало (а.е.)':<18} {'Конец (а.е.)':<18} {'Отклонение (%)':<18}")
    print("-" * 80)

    for i in range(1, data.n_particles):
        dist_initial = np.linalg.norm(positions_history[0, i] - sun_pos_initial) / AU
        dist_final = np.linalg.norm(positions_history[-1, i] - sun_pos_final) / AU
        deviation = abs(dist_final - dist_initial) / dist_initial * 100
        print(f"{planet_names[i]:<12} {dist_initial:>16.4f}   {dist_final:>16.4f}   {deviation:>16.6f}")

    print("-" * 80)

    # Энергия
    energy_error = abs(total_energies[-1] - total_energies[0]) / abs(total_energies[0]) * 100
    print(f"\n⚡ СОХРАНЕНИЕ ЭНЕРГИИ:")
    print(f"  Начальная полная энергия: {total_energies[0]:.6e} Дж")
    print(f"  Конечная полная энергия:  {total_energies[-1]:.6e} Дж")
    print(f"  Относительная ошибка:     {energy_error:.6e} %")
    print("=" * 80)

    # ========== 4. ВИЗУАЛИЗАЦИЯ ==========

    # График траекторий
    print("\n📈 Построение графика траекторий...")
    plotter = Plotter2D(
        positions_history=positions_history,
        masses=data.masses,
        dt=args.dt,
        particle_names=planet_names,
        particle_colors=planet_colors,
        title=f'🌌 Солнечная система ({args.years} {"год" if args.years == 1 else "лет"}, метод: {args.method.upper()})',
        dark_theme=True,
    )

    if args.save:
        filename = f'trajectory_solar_{args.method}_dt{int(args.dt)}_y{int(args.years)}.png'
        plotter.save(filename, dpi=300)
        print(f"  ✅ Сохранено: {filename}")

    plotter.visualize(show_start=True, show_end=False, show=True)

    # График энергии
    print("⚡ Построение графика энергии...")
    energy_plotter = EnergyPlotter(
        times=times / year_seconds,  # Переводим в годы
        kinetic_energies=kinetic_energies,
        potential_energies=potential_energies,
        total_energies=total_energies,
        title=f'⚡ Сохранение энергии (метод: {args.method.upper()}, dt={args.dt}s)',
        dark_theme=True,
        time_label='Time (years)'
    )

    if args.save:
        filename = f'energy_solar_{args.method}_dt{int(args.dt)}_y{int(args.years)}.png'
        energy_plotter.save(filename, dpi=300)
        print(f"  ✅ Сохранено: {filename}")

    energy_plotter.visualize(show_relative_error=True, show=True)

    # Дополнительный график: орбиты отдельных планет
    if not args.inner_only and args.years >= 10:
        print("🪐 Построение графика внешних планет...")
        # Индексы внешних планет: Jupiter (5), Saturn (6), Uranus (7), Neptune (8), Pluto (9)
        outer_indices = [0, 5, 6, 7, 8, 9]  # Включаем Солнце

        outer_positions = positions_history[:, outer_indices, :]
        outer_masses = data.masses[outer_indices]
        outer_names = [planet_names[i] for i in outer_indices]
        outer_colors = [planet_colors[i] for i in outer_indices]

        plotter_outer = Plotter2D(
            positions_history=outer_positions,
            masses=outer_masses,
            dt=args.dt,
            particle_names=outer_names,
            particle_colors=outer_colors,
            title=f'🪐 Внешние планеты ({args.years} лет, метод: {args.method.upper()})',
            dark_theme=True,
        )

        if args.save:
            filename = f'trajectory_outer_{args.method}_dt{int(args.dt)}_y{int(args.years)}.png'
            plotter_outer.save(filename, dpi=300)
            print(f"  ✅ Сохранено: {filename}")

        plotter_outer.visualize(show_start=False, show_end=False, show=True)

    # Анимация
    if not args.no_animation:
        print("\n🎬 Создание анимации...")

        # Определяем интервал для анимации (чтобы не было слишком медленно)
        # Показываем каждый N-й кадр
        if n_steps > 5000:
            frame_skip = n_steps // 5000
            print(f"  ℹ️  Пропуск кадров: показываем каждый {frame_skip}-й")
            anim_positions = positions_history[::frame_skip]
            anim_dt = args.dt * frame_skip
        else:
            anim_positions = positions_history
            anim_dt = args.dt

        animator = Animator2D(
            positions_history=anim_positions,
            masses=data.masses,
            dt=anim_dt,
            particle_names=planet_names,
            particle_colors=planet_colors,
            title=f'🌌 Солнечная система (метод: {args.method.upper()})',
            show_trails=True,
            trail_length=args.trail_length,
            dark_theme=True,
        )

        if args.save:
            print("  💾 Сохранение анимации (требует ffmpeg, это может занять время)...")
            filename = f'animation_solar_{args.method}_dt{int(args.dt)}_y{int(args.years)}.mp4'
            animator.save(filename, fps=30, dpi=150)
            print(f"  ✅ Сохранено: {filename}")

        print("  🎥 Запуск анимации (закройте окно для продолжения)...")
        animator.visualize(interval=20, repeat=True, show=True)

    # ========== 5. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ==========
    print("\n" + "=" * 80)
    print("📊 ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 80)

    # Расчёт орбитальных периодов
    print("\n🔄 ОРБИТАЛЬНЫЕ ПЕРИОДЫ:")
    print("-" * 80)
    print(f"{'Планета':<12} {'Реальный период (годы)':<25} {'Макс. расст. (а.е.)':<20}")
    print("-" * 80)

    # Известные периоды (для сравнения)
    known_periods = {
        'Mercury': 0.241,
        'Venus': 0.615,
        'Earth': 1.000,
        'Mars': 1.881,
        'Jupiter': 11.862,
        'Saturn': 29.457,
        'Uranus': 84.011,
        'Neptune': 164.79,
        'Pluto': 247.94,
    }

    sun_pos = positions_history[:, 0, :]

    for i in range(1, data.n_particles):
        # Расстояния от Солнца
        distances = np.linalg.norm(positions_history[:, i, :] - sun_pos, axis=1) / AU
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)

        planet_name = planet_names[i]
        real_period = known_periods.get(planet_name, 0.0)

        print(f"{planet_name:<12} {real_period:>23.3f}   {max_distance:>18.4f}")

    print("-" * 80)

    # Анализ устойчивости системы
    print("\n🎯 АНАЛИЗ УСТОЙЧИВОСТИ:")

    # Проверка столкновений (минимальные расстояния между планетами)
    min_distances = {}
    for i in range(1, data.n_particles):
        for j in range(i + 1, data.n_particles):
            distances = np.linalg.norm(
                positions_history[:, i, :] - positions_history[:, j, :],
                axis=1
            ) / AU
            min_dist = np.min(distances)
            min_distances[f"{planet_names[i]}-{planet_names[j]}"] = min_dist

    # Показываем 5 самых близких сближений
    sorted_distances = sorted(min_distances.items(), key=lambda x: x[1])
    print("\n  🔍 Топ-5 самых близких сближений:")
    print("  " + "-" * 60)
    for pair, dist in sorted_distances[:5]:
        print(f"  {pair:<30} {dist:>10.4f} а.е.")

    # Центр масс
    print("\n  🎯 Смещение центра масс:")
    total_mass = np.sum(data.masses)
    com_initial = np.sum(data.masses * positions_history[0], axis=0) / total_mass
    com_final = np.sum(data.masses * positions_history[-1], axis=0) / total_mass
    com_drift = np.linalg.norm(com_final - com_initial) / AU
    print(
        f"  Начальное положение ЦМ: ({com_initial[0] / AU:.6f}, {com_initial[1] / AU:.6f}, {com_initial[2] / AU:.6f}) а.е.")
    print(f"  Конечное положение ЦМ:  ({com_final[0] / AU:.6f}, {com_final[1] / AU:.6f}, {com_final[2] / AU:.6f}) а.е.")
    print(f"  Смещение ЦМ:            {com_drift:.6e} а.е.")

    # Момент импульса
    L_initial = np.sum(
        data.masses * np.cross(positions_history[0], velocities_history[0]),
        axis=0
    )
    L_final = np.sum(
        data.masses * np.cross(positions_history[-1], velocities_history[-1]),
        axis=0
    )
    L_error = np.linalg.norm(L_final - L_initial) / np.linalg.norm(L_initial) * 100

    print(f"\n  🌀 Сохранение момента импульса:")
    print(f"  Относительная ошибка: {L_error:.6e} %")

    print("=" * 80)

    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    if energy_error > 1.0:
        print("  ⚠️  Большая ошибка энергии! Рекомендуется:")
        print("     - Уменьшить временной шаг (--dt)")
        print("     - Использовать метод RK4 или LF (--method rk4/lf)")
    elif energy_error > 0.01:
        print("  ⚡ Умеренная ошибка энергии. Для лучшей точности:")
        print("     - Уменьшите временной шаг в 2 раза")
    else:
        print("  ✅ Отличное сохранение энергии!")

    if com_drift > 1e-6:
        print("  ⚠️  Заметное смещение центра масс - возможна численная ошибка")
    else:
        print("  ✅ Центр масс практически неподвижен")

    if L_error > 1.0:
        print("  ⚠️  Большая ошибка момента импульса!")
    else:
        print("  ✅ Хорошее сохранение момента импульса")

    print("\n" + "=" * 80)
    print("✨ СИМУЛЯЦИЯ ЗАВЕРШЕНА!")
    print("=" * 80)


if __name__ == '__main__':
    main()