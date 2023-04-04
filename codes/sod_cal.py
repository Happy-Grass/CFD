import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
delta_x = 0.01
delta_t = 0.01


def phi_r(U):
    # U -> 2-D array
    # Van Leer
    r = np.zeros_like(U)
    start, end = 1, -2
    r[:, start:end] = (U[:, start:end] - U[:, start - 1:end - 1]) / (
        U[:, start + 1:end + 1] - U[:, start:end])
    phir = (np.abs(r) + r) / (np.abs(r) + 1)
    return phir


def Uj12(U):
    start, end = 0, -2
    temp = np.zeros_like(U)
    temp[:, start:end] = U[:, start:end] + phi_r(U)[:, start:end] * (
        U[:, start + 1:end + 1] - U[:, start:end]) / 2
    return temp


def Uj_12(U):
    start, end = 1, -1
    temp = np.zeros_like(U)
    temp[:,
         start:end] = U[:,
                        start - 1:end - 1] + phi_r(U)[:, start - 1:end - 1] * (
                            U[:, start:end] - U[:, start - 1:end - 1]) / 2
    return temp


def p_x(U, delta_x):
    # 二阶TVD
    pu = (Uj12(U) - Uj_12(U)) / delta_x
    return pu


# calc A_matrix
def parf_u(U):
    # get the matrix A
    # for U = [u1, u2, u3, ..., un] -> A = [A1, A2, A3, ..., An]
    # where Ai = [[0, 1, 0], [, ,], [, , ]]
    m, n = U.shape
    a1, a2, a3 = U[0, :], U[1, :], U[2, :]
    zero = np.zeros(shape=(n, ))
    one = np.ones(shape=(n, ))
    A = np.zeros(shape=(m, 3 * n))
    A[:, 0::3] = np.array([
        one, (gamma - 3) / 2 * a2 * a2 / a1 / a1,
        -gamma * a2 * a3 / a1 / a1 + (gamma - 1) * a3 * a3 * a3 / a1 / a1 / a1
    ])
    A[:, 1::3] = np.array([
        one, (3 - gamma) * a2 / a1,
        gamma * a3 / a1 - 3 * (gamma - 1) * a2 * a2 / a1 / a1
    ])
    A[:, 2::3] = np.array([zero, gamma - one, gamma * a2 / a1])
    return A


def h_u(U, delta_x):
    # 待优化
    _, n = U.shape
    A = parf_u(U)
    pu = p_x(U, delta_x)
    hu = np.zeros_like(U)
    for i in range(n):
        hu[:, i] = np.matmul(A[:, 3 * i:3 * i + 3], pu[:, i])
    return -hu


def step_time(U, delta_x, delta_t):
    U1 = U + delta_t * h_u(U, delta_x)
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * delta_t * h_u(U1, delta_x)
    U_next = 1 / 3 * U + 2 / 3 * U2 + 2 / 3 * delta_t * h_u(U2, delta_x)
    return U_next


# print(u[:, 0::3].shape)
# a1, a2, a3 = u[0, :], u[1, :], u[2, :]
# print(np.zeros(shape=(7, )))
# print(a1 * a1)
# print(parf_u(u).shape)

# print(parf_u(u))


def func_rho(x):
    temp = x.copy()
    temp[temp < 0.5] = 1
    temp[temp >= 0.5] = 0.125
    return temp


def func_u(x):
    temp = x.copy()
    temp[temp < 0.5] = 0
    temp[temp >= 0.5] = 0
    return temp


def func_p(x):
    temp = x.copy()
    temp[temp < 0.5] = 1
    temp[temp >= 0.5] = 0.1
    return temp


def init():
    x = np.linspace(0, 1, 101, endpoint=True)
    u = func_u(x)
    rho = func_rho(x)
    p = func_p(x)
    rho_u = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u * u
    U = np.array([rho, rho_u, E])
    return U


U = init()
U_next = step_time(U, delta_x, delta_t)
print(U_next)
