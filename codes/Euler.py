# /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def partial_v(u, dv, methods="Euler"):
    partialu_v = np.zeros_like(u)
    if methods == "Euler":
        # forward difference method
        start, end = 1, -1
        partialu_v[start:end] = (u[start:end] - u[start - 1:end - 1]) / dv
        return partialu_v
    else:
        print(
            "The method has not been defined, please enter the correct method!"
        )
        return None


def fu(u):
    return 0


def lu_def(u, func, var_a, partialu_v, dv, methods="Euler"):
    return func(u) - var_a * partialu_v(u, dv, methods=methods)


def time_step(u_cur, lu, dt, methods="Euler"):
    if methods == "Euler":
        u_next = u_cur + dt * lu(u_cur)
    elif methods == "2rk":
        u1 = u_cur + dt * lu(u_cur)
        u_next = 0.5 * u_cur + 0.5 * u1 + 0.5 * dt * lu(u1)
    elif methods == "3rk":
        u1 = u_cur + dt * lu(u_cur)
        u2 = 0.75 * u_cur + 0.25 * (u1 + dt * lu(u1))
        u_next = 1 / 3 * u_cur + 2 / 3 * (u2 + dt * lu(u2))
    elif methods == "4rk":
        u1 = u_cur + 0.5 * dt * lu(u_cur)
        u2 = u_cur + 0.5 * dt * lu(u1)
        u3 = u_cur + dt * lu(u2)
        u_next = 1 / 3 + (-u_cur + u1 + 2 * u2 + u3) + 1 / 6 * dt * lu(u3)
    else:
        print(
            "The method has not been defined, please enter the correct method")
        return None
    return u_next


x = np.linspace(0, 1, 101)
u = np.sin(x)
dx = 1 / 100
dt = 0.01
for i in range(1000):
    partialu_x = partial_v(u, dx)
    func_u = fu(u)
    lu_v = lu(func_u, var_a=1, partialu_v=partialu_x)
    u = time_step(u, lu_v, dt)
plt.plot(x, u)
plt.show()
