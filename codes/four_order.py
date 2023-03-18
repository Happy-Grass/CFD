import numpy as np
import matplotlib.pyplot as plt


DELTA_T = 0.01
DELTA_X = 0.1 * np.pi


# the coeffient of u_j
a_1 = -1 / 540 / DELTA_X
a_2 = -5 * a_1 + 1 / 12 / DELTA_X
a_3 = 10 * a_1 - 2 / 3 / DELTA_X
a_4 = -10 * a_1
a_5 = 5 * a_1 + 2 / 3 / DELTA_X
a_6 = -a_1 - 1 / 12 / DELTA_X
coeffient = np.array([a_1, a_2, a_3, a_4, a_5, a_6])


def L_U(U, coeffient):
    (scope,) = coeffient.shape
    (length,) = U.shape
    L_U_Res = np.zeros(length - 5)
    for i in range(0, length - 5):
        L_U_Res[i] = np.dot(U[i : i + scope : 1], coeffient)
    return L_U_Res


def U_n_next(U_n):
    U_1 = -1 * L_U(U_n, coeffient) * DELTA_T + U_n[3:-2]
    U_2 = 0.75 * U_n[6:-4] + 0.25 * (U_1[3:-2] - L_U(U_1, coeffient) * DELTA_T)
    U_n1 = 1 / 3 * U_n[9:-6] + 2 / 3 * (U_2[3:-2] - L_U(U_2, coeffient) * DELTA_T)
    return U_n1


def init_func(x):
    return np.sin(x)


def calc_u_t(t, init_func):
    t_length = 1 + int(t / DELTA_T)
    t_step = np.linspace(0, t, t_length, endpoint=True)
    x_min = 0 - 9 * DELTA_X * (t_length - 1)
    x_max = 2 * np.pi + 6 * DELTA_X * (t_length - 1)
    x = np.linspace(x_min, x_max, int((x_max - x_min)/DELTA_X)+1)
    u_x_0 = init_func(x)
    for t in t_step[1:]:
        u_x_0 = U_n_next(u_x_0)
        print("Calculating the value when t: {}".format(t))
    return u_x_0


def true_value(x, t):
    return np.sin(x - t)


# result plotting
t = 50
x = np.arange(0, 2 * np.pi + DELTA_X, DELTA_X)
u_x_t = calc_u_t(t, init_func)
truevalue = true_value(x, t)
print(truevalue.shape)

L2_error = np.square(truevalue - u_x_t).sum()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, truevalue, "-", label="True Value")
ax.plot(x, u_x_t, "-o", label="Calculated Value")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.text(
    0.05,
    0.9,
    ("$t={}$" + "\n" + "$Error(L_2)={:.5f}$").format(t, L2_error),
    transform=ax.transAxes,
)
ax.legend(loc="upper right", frameon=False)
fig.savefig("t50.png", dpi=500)
