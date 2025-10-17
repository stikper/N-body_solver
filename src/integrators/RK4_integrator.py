

class RK4Integrator:
    @staticmethod
    def step(f, t, y, dt):
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt,   y + dt * k3)
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)