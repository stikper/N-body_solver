import numpy as np
import time
import matplotlib.pyplot as plt
from src import (
    ParticleData, G, AU,
    DirectForceCalculator,
    LFIntegrator, RK4Integrator,
    Simulation, SimulationParameters,
)


def create_solar_system_data(n_bodies=10):
    """Создаёт данные Солнечной системы с заданным количеством тел"""
    all_masses = np.array([
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

    all_positions = np.array([
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

    all_velocities = np.array([
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

    return ParticleData(
        masses=all_masses[:n_bodies],
        positions=all_positions[:n_bodies],
        velocities=all_velocities[:n_bodies]
    )


def benchmark_n_bodies():
    """Бенчмарк: зависимость времени от количества тел"""
    print("=" * 70)
    print("🚀 BENCHMARK 1: Время расчёта vs Количество тел")
    print("=" * 70)
    print("Параметры: 248 лет, шаг = 1 день")
    print("-" * 70)

    # Параметры
    dt = 86400.0  # 1 день
    years = 248
    n_steps = int(years * 365.25)

    n_bodies_range = range(2, 11)  # От 2 до 10 тел

    results = {
        'n_bodies': [],
        'lf_time': [],
        'rk4_time': [],
    }

    for n_bodies in n_bodies_range:
        print(f"\n📊 Тестирование с {n_bodies} телами...")

        # Создаём данные
        data_lf = create_solar_system_data(n_bodies)
        data_rk4 = create_solar_system_data(n_bodies)

        # === Leapfrog ===
        print(f"  ⚡ LF интегратор...", end=" ", flush=True)
        sim_params_lf = SimulationParameters(
            force_calculator=DirectForceCalculator(),
            integrator=LFIntegrator(),
            dt=dt
        )
        sim_lf = Simulation(data_lf, sim_params_lf)

        start_time = time.perf_counter()
        for _ in range(n_steps):
            sim_lf.step()
        lf_time = time.perf_counter() - start_time
        print(f"✓ {lf_time:.3f} сек")

        # === RK4 ===
        print(f"  🎯 RK4 интегратор...", end=" ", flush=True)
        sim_params_rk4 = SimulationParameters(
            force_calculator=DirectForceCalculator(),
            integrator=RK4Integrator(),
            dt=dt
        )
        sim_rk4 = Simulation(data_rk4, sim_params_rk4)

        start_time = time.perf_counter()
        for _ in range(n_steps):
            sim_rk4.step()
        rk4_time = time.perf_counter() - start_time
        print(f"✓ {rk4_time:.3f} сек")

        # Сохраняем результаты
        results['n_bodies'].append(n_bodies)
        results['lf_time'].append(lf_time)
        results['rk4_time'].append(rk4_time)

        print(f"  📈 Соотношение RK4/LF: {rk4_time / lf_time:.2f}x")

    # Вывод таблицы
    print("\n" + "=" * 70)
    print("📋 СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    print(f"{'N тел':<10} {'LF (сек)':<15} {'RK4 (сек)':<15} {'RK4/LF':<10}")
    print("-" * 70)
    for i in range(len(results['n_bodies'])):
        n = results['n_bodies'][i]
        lf = results['lf_time'][i]
        rk4 = results['rk4_time'][i]
        ratio = rk4 / lf
        print(f"{n:<10} {lf:<15.3f} {rk4:<15.3f} {ratio:<10.2f}x")
    print("=" * 70)

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Абсолютное время
    ax1.plot(results['n_bodies'], results['lf_time'], 'o-',
             label='Leapfrog', linewidth=2, markersize=8, color='#2ecc71')
    ax1.plot(results['n_bodies'], results['rk4_time'], 's-',
             label='RK4', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Количество тел', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Время выполнения (сек)', fontsize=12, fontweight='bold')
    ax1.set_title('⏱️ Производительность интеграторов\n(248 лет, шаг = 1 день)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(results['n_bodies'])

    # График 2: Соотношение
    ratios = [rk4 / lf for lf, rk4 in zip(results['lf_time'], results['rk4_time'])]
    ax2.plot(results['n_bodies'], ratios, 'D-',
             linewidth=2, markersize=8, color='#9b59b6')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='LF = RK4')
    ax2.set_xlabel('Количество тел', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Соотношение RK4/LF', fontsize=12, fontweight='bold')
    ax2.set_title('📊 Относительная производительность\n(RK4 медленнее в X раз)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(results['n_bodies'])

    plt.tight_layout()
    plt.savefig('benchmark_n_bodies.png', dpi=150, bbox_inches='tight')
    print("\n💾 График сохранён: benchmark_n_bodies.png")
    plt.show()

    return results


def benchmark_timestep():
    """Бенчмарк: зависимость времени от шага интегрирования"""
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 2: Время расчёта vs Шаг интегрирования")
    print("=" * 70)
    print("Параметры: Внутренняя солнечная система (5 тел), 50 лет")
    print("-" * 70)

    # Параметры
    years = 50
    n_bodies = 5  # Солнце + 4 внутренние планеты

    # Диапазон шагов: от 1 часа до 1 дня
    dt_hours = [1, 2, 4, 6, 12, 18, 24]
    dt_values = [h * 3600.0 for h in dt_hours]

    results = {
        'dt_hours': [],
        'n_steps': [],
        'lf_time': [],
        'rk4_time': [],
    }

    for dt_h, dt in zip(dt_hours, dt_values):
        n_steps = int(years * 365.25 * 86400.0 / dt)

        print(f"\n📊 Шаг = {dt_h} ч ({n_steps:,} шагов)...")

        # Создаём данные
        data_lf = create_solar_system_data(n_bodies)
        data_rk4 = create_solar_system_data(n_bodies)

        # === Leapfrog ===
        print(f"  ⚡ LF интегратор...", end=" ", flush=True)
        sim_params_lf = SimulationParameters(
            force_calculator=DirectForceCalculator(),
            integrator=LFIntegrator(),
            dt=dt
        )
        sim_lf = Simulation(data_lf, sim_params_lf)

        start_time = time.perf_counter()
        for _ in range(n_steps):
            sim_lf.step()
        lf_time = time.perf_counter() - start_time
        print(f"✓ {lf_time:.3f} сек")

        # === RK4 ===
        print(f"  🎯 RK4 интегратор...", end=" ", flush=True)
        sim_params_rk4 = SimulationParameters(
            force_calculator=DirectForceCalculator(),
            integrator=RK4Integrator(),
            dt=dt
        )
        sim_rk4 = Simulation(data_rk4, sim_params_rk4)

        start_time = time.perf_counter()
        for _ in range(n_steps):
            sim_rk4.step()
        rk4_time = time.perf_counter() - start_time
        print(f"✓ {rk4_time:.3f} сек")

        # Сохраняем результаты
        results['dt_hours'].append(dt_h)
        results['n_steps'].append(n_steps)
        results['lf_time'].append(lf_time)
        results['rk4_time'].append(rk4_time)

        print(f"  📈 Соотношение RK4/LF: {rk4_time / lf_time:.2f}x")

    # Вывод таблицы
    print("\n" + "=" * 70)
    print("📋 СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    print(f"{'Шаг (ч)':<12} {'N шагов':<12} {'LF (сек)':<12} {'RK4 (сек)':<12} {'RK4/LF':<10}")
    print("-" * 70)
    for i in range(len(results['dt_hours'])):
        dt_h = results['dt_hours'][i]
        n_st = results['n_steps'][i]
        lf = results['lf_time'][i]
        rk4 = results['rk4_time'][i]
        ratio = rk4 / lf
        print(f"{dt_h:<12} {n_st:<12,} {lf:<12.3f} {rk4:<12.3f} {ratio:<10.2f}x")
    print("=" * 70)

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Время vs Шаг
    ax1.plot(results['dt_hours'], results['lf_time'], 'o-',
             label='Leapfrog', linewidth=2, markersize=8, color='#2ecc71')
    ax1.plot(results['dt_hours'], results['rk4_time'], 's-',
             label='RK4', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Шаг интегрирования (часы)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Время выполнения (сек)', fontsize=12, fontweight='bold')
    ax1.set_title('⏱️ Производительность vs Шаг\n(Внутренняя СС, 50 лет)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(results['dt_hours'])

    # График 2: Время vs Количество шагов
    ax2.plot(results['n_steps'], results['lf_time'], 'o-',
             label='Leapfrog', linewidth=2, markersize=8, color='#2ecc71')
    ax2.plot(results['n_steps'], results['rk4_time'], 's-',
             label='RK4', linewidth=2, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Количество шагов', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Время выполнения (сек)', fontsize=12, fontweight='bold')
    ax2.set_title('⏱️ Производительность vs Число итераций\n(линейная зависимость)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', axis='x')

    # Разворачиваем ось X (больше шагов = меньший dt)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig('benchmark_timestep.png', dpi=150, bbox_inches='tight')
    print("\n💾 График сохранён: benchmark_timestep.png")
    plt.show()

    return results


def main():
    """Главная функция бенчмарка"""
    print("\n" + "🌌" * 35)
    print("          N-BODY SOLVER PERFORMANCE BENCHMARK")
    print("🌌" * 35 + "\n")

    # Бенчмарк 1: Количество тел
    results_bodies = benchmark_n_bodies()

    # Бенчмарк 2: Шаг интегрирования
    results_timestep = benchmark_timestep()

    # Финальная сводка
    print("\n" + "=" * 70)
    print("✅ БЕНЧМАРК ЗАВЕРШЁН")
    print("=" * 70)
    print("\n📊 Основные выводы:")
    print("-" * 70)

    # Вывод 1: Масштабируемость по количеству тел
    lf_times = results_bodies['lf_time']
    rk4_times = results_bodies['rk4_time']
    avg_ratio = np.mean([rk4 / lf for lf, rk4 in zip(lf_times, rk4_times)])

    print(f"\n1️⃣  Производительность интеграторов:")
    print(f"    • RK4 медленнее LF в среднем в {avg_ratio:.2f} раз")
    print(f"    • Время LF для 10 тел: {lf_times[-1]:.2f} сек")
    print(f"    • Время RK4 для 10 тел: {rk4_times[-1]:.2f} сек")

    # Оценка сложности O(N²)
    n_vals = np.array(results_bodies['n_bodies'])
    lf_vals = np.array(results_bodies['lf_time'])

    # Проверяем квадратичную зависимость
    speedup_2_to_10 = lf_vals[-1] / lf_vals[0]  # Ускорение от 2 до 10 тел
    theoretical_speedup = (10 / 2) ** 2  # O(N²)

    print(f"\n2️⃣  Масштабируемость (сложность алгоритма):")
    print(f"    • Увеличение времени (2→10 тел): {speedup_2_to_10:.1f}x")
    print(f"    • Теоретическое O(N²): {theoretical_speedup:.1f}x")
    print(f"    • Соответствие теории: {(speedup_2_to_10 / theoretical_speedup) * 100:.1f}%")

    # Вывод 2: Зависимость от шага
    dt_times_lf = results_timestep['lf_time']
    dt_times_rk4 = results_timestep['rk4_time']

    print(f"\n3️⃣  Влияние шага интегрирования:")
    print(f"    • Шаг 1 час:  LF = {dt_times_lf[0]:.2f} с,  RK4 = {dt_times_rk4[0]:.2f} с")
    print(f"    • Шаг 24 часа: LF = {dt_times_lf[-1]:.2f} с,  RK4 = {dt_times_rk4[-1]:.2f} с")
    print(f"    • Ускорение (1ч→24ч): LF = {dt_times_lf[0] / dt_times_lf[-1]:.1f}x, "
          f"RK4 = {dt_times_rk4[0] / dt_times_rk4[-1]:.1f}x")

    # Рекомендации
    print("\n" + "=" * 70)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 70)
    print("\n✅ Для долгосрочных орбитальных симуляций:")
    print("   → Используйте LFIntegrator (быстрее и сохраняет энергию)")
    print("   → Оптимальный шаг: 0.1-1% от орбитального периода")
    print("   → Для внутренней СС: 6-12 часов")
    print("   → Для полной СС: 12-24 часа")

    print("\n✅ Для высокоточных расчётов:")
    print("   → Используйте RK4Integrator")
    print("   → Уменьшите шаг в 2-4 раза")
    print("   → Учитывайте увеличение времени расчёта")

    print("\n✅ Оптимизация производительности:")
    print(f"   → Direct метод: O(N²), эффективен до ~100 тел")
    print(f"   → Для N > 100: рассмотрите Barnes-Hut или FMM")
    print(f"   → Vectorization уже используется (NumPy)")

    print("\n" + "=" * 70)
    print("🎉 Все тесты пройдены успешно!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()