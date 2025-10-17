import numpy as np
import matplotlib.pyplot as plt

from src.integrators.RK4_integrator import RK4Integrator

def f(_, y):
    return y # dy/dt = y


t0, y0 = 0.0, 1.0
t_max = 5.0
dt = 1
dt_analytic = 0.01 # Less than dt to get higher resolution

times = np.arange(t0, t_max + dt, dt)
times_analytic = np.arange(t0, t_max + dt_analytic, dt_analytic)
y_num = np.zeros_like(times)
y_num[0] = y0

for i in range(1, len(times)):
    y_num[i] = RK4Integrator.step(f, times[i-1], y_num[i-1], dt)

y_analytic = np.exp(times_analytic) # Analytic solution of eqn

plt.figure(figsize=(7, 4))
plt.plot(times_analytic, y_analytic, 'r--', label='Analytical $e^t$')
plt.plot(times, y_num, 'b-o', label='RK4 numerical')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Test of RK4 Integrator: dy/dt = y')
plt.legend()
plt.grid(True)
plt.show()