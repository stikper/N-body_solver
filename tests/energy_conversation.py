import numpy as np
import matplotlib.pyplot as plt
from src import (
    ParticleData,
    DirectForceCalculator,
    LFIntegrator, RK4Integrator,
    Simulation, SimulationParameters,
)


def create_solar_system_data():
    """Создаёт данные Солнечной системы"""
    masses = np.array([
        [1.98841e30],  # Sun
        [3.30110e23],  # Mercury
        [4.86732e24],  # Venus
        [5.97217e24],  # Earth
        [6.41693e23],  # Mars
        [1.89813e27],  # Jupiter
        [5.68319e26],  # Saturn
        [8.68103e25],  # Uranus
        [1.02410e26],  # Neptune
        [1.30900e22],  # Pluto
    ])

    positions = np.array([
        [-479120095.480, -767705664.734, -310948056.613],
        [-58637939612.900, -4677510733.422, 3628042236.444],
        [-42523165514.879, -92973444841.244, -39140953872.635],
        [22407886361.781, 132721917099.621, 57554472840.542],
        [10087995256.084, -197521743479.253, -90842337693.048],
        [-233468076149.888, 680064632688.194, 297183625263.336],
        [1423015591043.480, 42942502046.692, -43554480707.443],
        [1487224413928.140, 2304369453427.526, 988215836744.449],
        [4468555127925.526, 103836004405.415, -68749496835.001],
        [2868338108400.359, -3932184130808.958, -2091335882359.679],
    ])

    velocities = np.array([
        [12.513, 0.097, -0.214],
        [-8431.777, -41541.575, -21316.719],
        [32058.118, -11778.176, -7327.505],
        [-29899.062, 4154.016, 1801.674],
        [25129.723, 3213.343, 796.159],
        [-12617.943, -3146.411, -1041.450],
        [-683.173, 8898.142, 3704.785],
        [-5907.321, 2861.175, 1336.658],
        [-118.393, 5059.455, 2073.811],
        [4716.140, 2213.847, -730.083],
    ])

    return ParticleData(masses=masses, positions=positions, velocities=velocities)


def run_simulation(method_name, dt, n_years=248):
    """Запуск симуляции с заданным методом"""
    print(f"\n{'=' * 60}")
    print(f"🚀 Запуск симуляции: {method_name.upper()}")
    print(f"{'=' * 60}")
    print(f"⏱️  Временной шаг: {dt} с ({dt / 86400:.2f} дней)")
    print(f"📅 Длительность: {n_years} лет")

    # Создаём данные
    data = create_solar_system_data()

    # Выбираем интегратор
    if method_name == 'rk4':
        integrator = RK4Integrator()
    else:
        integrator = LFIntegrator()

    # Параметры
    sim_params = SimulationParameters(
        force_calculator=DirectForceCalculator(),
        integrator=integrator,
        dt=dt
    )

    # Создаём симуляцию
    sim = Simulation(data=data, sim_params=sim_params)

    # Количество шагов
    year_seconds = 365.25 * 86400.0
    total_seconds = n_years * year_seconds
    n_steps = int(total_seconds / dt)

    print(f"🔄 Количество шагов: {n_steps:,}")

    # Массивы для энергии
    times = np.zeros(n_steps)
    total_energies = np.zeros(n_steps)

    # Главный цикл
    print("⏳ Симуляция...\n")
    for i in range(n_steps):
        times[i] = sim.time
        total_energies[i] = sim.system.calc_total_energy()
        sim.step()

        # Прогресс
        if i % (n_steps // 20) == 0:
            progress = 100 * i / n_steps
            print(f"  {progress:5.1f}% | Шаг {i:>10,}/{n_steps:,}")

    print("\n✅ Готово!")

    return times, total_energies


def plot_comparison(times_rk4, energies_rk4, times_lf, energies_lf, dt):
    """Построение графиков сравнения"""

    # Переводим время в годы
    year_seconds = 365.25 * 86400.0
    times_rk4_years = times_rk4 / year_seconds
    times_lf_years = times_lf / year_seconds

    # Относительные ошибки
    E0_rk4 = energies_rk4[0]
    E0_lf = energies_lf[0]

    rel_error_rk4 = (energies_rk4 - E0_rk4) / abs(E0_rk4) * 100
    rel_error_lf = (energies_lf - E0_lf) / abs(E0_lf) * 100

    # Создаём фигуру с 2 графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#0E1117')

    # ========== График 1: Полная энергия ==========
    ax1.set_facecolor('#1E1E1E')

    ax1.plot(times_rk4_years, energies_rk4,
             color='#FF6B6B', linewidth=1.5, label='RK4', alpha=0.9)
    ax1.plot(times_lf_years, energies_lf,
             color='#4ECDC4', linewidth=1.5, label='Leap-Frog', alpha=0.9)

    ax1.set_xlabel('Время (годы)', fontsize=12, color='white')
    ax1.set_ylabel('Полная энергия (Дж)', fontsize=12, color='white')
    ax1.set_title(f'⚡ Полная энергия системы (dt = {dt / 86400:.1f} дней)',
                  fontsize=14, color='white', pad=20)
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white')

    # Цвет осей
    for spine in ax1.spines.values():
        spine.set_color('white')

    # ========== График 2: Относительная ошибка ==========
    ax2.set_facecolor('#1E1E1E')

    ax2.plot(times_rk4_years, rel_error_rk4,
             color='#FF6B6B', linewidth=1.5, label='RK4', alpha=0.9)
    ax2.plot(times_lf_years, rel_error_lf,
             color='#4ECDC4', linewidth=1.5, label='Leap-Frog', alpha=0.9)

    ax2.set_xlabel('Время (годы)', fontsize=12, color='white')
    ax2.set_ylabel('Относительная ошибка энергии (%)', fontsize=12, color='white')
    ax2.set_title('📊 Относительная ошибка сохранения энергии',
                  fontsize=14, color='white', pad=20)
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.2, color='white')
    ax2.tick_params(colors='white')
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)

    # Цвет осей
    for spine in ax2.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    # Сохранение
    filename = f'energy_comparison_dt{int(dt)}.png'
    plt.savefig(filename, dpi=300, facecolor='#0E1117')
    print(f"\n💾 График сохранён: {filename}")

    plt.show()


def print_statistics(times_rk4, energies_rk4, times_lf, energies_lf, dt):
    """Вывод статистики"""
    year_seconds = 365.25 * 86400.0

    # Ошибки
    E0_rk4 = energies_rk4[0]
    E0_lf = energies_lf[0]

    error_rk4 = abs(energies_rk4[-1] - E0_rk4) / abs(E0_rk4) * 100
    error_lf = abs(energies_lf[-1] - E0_lf) / abs(E0_lf) * 100

    max_error_rk4 = np.max(np.abs(energies_rk4 - E0_rk4)) / abs(E0_rk4) * 100
    max_error_lf = np.max(np.abs(energies_lf - E0_lf)) / abs(E0_lf) * 100

    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА СРАВНЕНИЯ МЕТОДОВ")
    print("=" * 70)
    print(f"\n⏱️  Временной шаг: {dt} с ({dt / 86400:.2f} дней)")
    print(f"📅 Длительность: {times_rk4[-1] / year_seconds:.1f} лет")
    print(f"🔄 Количество шагов: {len(times_rk4):,}")

    print(f"\n{'=' * 70}")
    print(f"{'Метод':<20} {'Конечная ошибка (%)':<25} {'Макс. ошибка (%)':<25}")
    print(f"{'=' * 70}")
    print(f"{'RK4':<20} {error_rk4:>23.6e}   {max_error_rk4:>23.6e}")
    print(f"{'Leap-Frog':<20} {error_lf:>23.6e}   {max_error_lf:>23.6e}")
    print(f"{'=' * 70}")

    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print("-" * 70)

    if error_rk4 < error_lf:
        print("🏆 RK4 показал лучшую точность!")
        print("   ✅ Используйте RK4 для краткосрочных симуляций (1-10 лет)")
        print("   ✅ Высокая точность, но медленнее")
    else:
        print("🏆 Leap-Frog показал лучшую точность!")
        print("   ✅ Используйте LF для долгосрочных симуляций (>10 лет)")
        print("   ✅ Сохраняет энергию лучше на длинных интервалах")

    print("\n📈 ОБЩИЕ ВЫВОДЫ:")
    if max_error_rk4 < 0.01 and max_error_lf < 0.01:
        print("   ✅ Оба метода показали отличную точность!")
    elif max_error_rk4 < 1.0 and max_error_lf < 1.0:
        print("   ⚡ Хорошая точность для обоих методов")
    else:
        print("   ⚠️  Рекомендуется уменьшить временной шаг")

    print("\n⏱️  СКОРОСТЬ:")
    print("   • RK4: 4 вычисления силы на шаг (медленнее)")
    print("   • Leap-Frog: 1 вычисление силы на шаг (быстрее)")

    print("\n🎯 КОГДА ЧТО ИСПОЛЬЗОВАТЬ:")
    print("   • Краткосрок (<10 лет): RK4 или LF - оба хороши")
    print("   • Долгосрок (>10 лет): LF - лучше сохраняет энергию")
    print("   • Очень долго (>100 лет): LF - симплектический метод")
    print("=" * 70)


def main():
    print("=" * 70)
    print("🌌 СРАВНЕНИЕ МЕТОДОВ ИНТЕГРИРОВАНИЯ")
    print("   Солнечная система, 248 лет (полный оборот Плутона)")
    print("=" * 70)

    # Параметры
    dt = 86400.0  # 1 день
    n_years = 2480

    # Запуск RK4
    times_rk4, energies_rk4 = run_simulation('rk4', dt, n_years)

    # Запуск Leap-Frog
    times_lf, energies_lf = run_simulation('lf', dt, n_years)

    # Статистика
    print_statistics(times_rk4, energies_rk4, times_lf, energies_lf, dt)

    # Графики
    print("\n📊 Построение графиков...")
    plot_comparison(times_rk4, energies_rk4, times_lf, energies_lf, dt)

    print("\n✨ Готово!")


if __name__ == '__main__':
    main()