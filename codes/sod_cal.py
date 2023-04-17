import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
delta_x = 0.01
delta_t = 0.01


def phi_r(U):
    # U -> 2-D array
    # Van Leer
    # U = np.array([[0.0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1],
    # [0, 0, 0, 0, 1, 1, 1, 1]])
    r = np.zeros_like(U)
    start, end = 1, -1
    duj1 = U[:, start + 1:] - U[:, start:end]
    duj = U[:, start:end] - U[:, start - 1:end - 1]
    r[:, start:end] = np.divide(duj,
                                duj1,
                                out=np.ones_like(duj) * 1000000.0,
                                where=duj1 != 0)
    print(r)
    phir = (np.abs(r) + r) / (np.abs(r) + 1)
    print(phir)
    return phir


def U12(U):
    start, end = 0, -1
    temp = U.copy()
    temp[:, start:end] = U[:, start:end] + phi_r(U)[:, start:end] * (
        U[:, start + 1:] - U[:, start:end]) / 2
    return temp


def U_12(U):
    start, end = 1, 0
    temp = U.copy()
    temp[:,
         start:] = U[:, start - 1:end - 1] + phi_r(U)[:, start - 1:end - 1] * (
             U[:, start:] - U[:, start - 1:end - 1]) / 2
    return temp


def p_x(U, delta_x):
    # 二阶TVD
    u_n, u_p = U12(U), U_12(U)
    pu = (u_n - u_p) / delta_x
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
        zero, (gamma - 3) / 2 * a2 * a2 / a1 / a1,
        -gamma * a2 * a3 / a1 / a1 + (gamma - 1) * a2 * a2 * a2 / a1 / a1 / a1
    ])
    A[:, 1::3] = np.array([
        one, (3 - gamma) * a2 / a1,
        gamma * a3 / a1 - 3 * (gamma - 1) * a2 * a2 / a1 / a1 / 2
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


def func_rho(x):
    temp = np.zeros_like(x)
    temp[x < 0.5] = 1
    temp[x >= 0.5] = 0.125
    return temp


def func_u(x):
    temp = np.zeros_like(x)
    temp[x < 0.5] = 0
    temp[x >= 0.5] = 0
    return temp


def func_p(x):
    temp = np.zeros_like(x)
    temp[x < 0.5] = 1
    temp[x >= 0.5] = 0.1
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


def update():
    x = np.linspace(0, 1, 101, endpoint=True)
    U = init()
    # U = np.array([np.sin(x + 0.1), np.sin(x), np.sin(x)])
    U_next = step_time(U, delta_x, delta_t)

    # for _ in range(100):
    #     U_next = step_time(U_next, delta_x, delta_t)
    # U_next = step_time(U_next, delta_x, delta_t)
    print(U_next)
    plt.plot(x, U[0, :], label='t')
    plt.plot(x, U_next[0, :], label='tnext')
    plt.legend()
    plt.show()


update()
