import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
delta_x = 0.01
delta_t = 0.001


def phi_r(f, positive: bool):
    # f <- 2-D array
    # Van Leer
    # f = np.array([[0.0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1],
    # [0, 0, 0, 0, 1, 1, 1, 1]])
    # for f_pos, positive=True; for f_neg, positive=False
    r = np.zeros_like(f)
    start, end = 1, -1
    fj1 = f[:, start + 1:] - f[:, start:end]
    fj = f[:, start:end] - f[:, start - 1:end - 1]
    if positive:
        r[:, start:end] = np.divide(fj,
                                    fj1,
                                    out=np.ones_like(fj1) * 1000000.0,
                                    where=fj1 != 0)
    else:
        r[:, start:end] = np.divide(fj1,
                                    fj,
                                    out=np.ones_like(fj) * 1000000.0,
                                    where=fj != 0)
    phir = (np.abs(r) + r) / (np.abs(r) + 1)
    return phir


def f12(f, positive):
    start, end = 0, -1
    calf = f.copy()
    if positive:
        calf[:,
             start:end] = f[:,
                            start:end] + phi_r(f, positive)[:, start:end] * (
                                f[:, start + 1:] - f[:, start:end]) * 0.5
    else:
        calf[:,
             start:end] = f[:,
                            start + 1:] + phi_r(f, positive)[:, start + 1:] * (
                                f[:, start:end] - f[:, start + 1:]) * 0.5
    return calf


def f_12(f, positive):
    start, end = 1, 0
    calf = f.copy()
    if positive:
        calf[:, start:] = f[:, start - 1:end -
                            1] + phi_r(f, positive)[:, start - 1:end - 1] * (
                                f[:, start:] - f[:, start - 1:end - 1]) * 0.5
    else:
        calf[:, start:] = f[:, start:] + phi_r(f, positive)[:, start:] * (
            f[:, start - 1:end - 1] - f[:, start:]) * 0.5
    return calf


def p_x(f, positive: bool, delta_x: float):
    # 二阶TVD
    f_n, f_p = f12(f, positive), f_12(f, positive)
    px = (f_n - f_p) / delta_x
    return px


def fvs(U):
    rho = U[0, :]
    u = U[1, :] / U[0, :]
    p = (gamma - 1) * (U[2, :] - 0.5 * rho * u * u)
    c = np.sqrt(gamma * p / rho)
    Lambda = np.array([u, u - c, u + c])
    lam_pos = 0.5 * Lambda + 0.5 * np.abs(Lambda)
    lam_neg = 0.5 * Lambda - 0.5 * np.abs(Lambda)
    f_pos = 0.5 * rho / gamma * np.array([
        2 * (gamma - 1) * lam_pos[0, :] + lam_pos[1, :] + lam_pos[2, :], 2 *
        (gamma - 1) * lam_pos[0, :] * Lambda[0, :] +
        lam_pos[1, :] * Lambda[1, :] + lam_pos[2, :] * Lambda[2, :],
        (gamma - 1) * lam_pos[0, :] * Lambda[0, :] * Lambda[0, :] +
        0.5 * lam_pos[1, :] * Lambda[1, :] * Lambda[1, :] +
        0.5 * lam_pos[2, :] * Lambda[2, :] * Lambda[2, :] + 0.5 * (3 - gamma) /
        (gamma - 1) * (lam_pos[1, :] + lam_pos[2, :]) * c * c
    ])

    f_neg = 0.5 * rho / gamma * np.array([
        2 * (gamma - 1) * lam_neg[0, :] + lam_neg[1, :] + lam_neg[2, :], 2 *
        (gamma - 1) * lam_neg[0, :] * Lambda[0, :] +
        lam_neg[1, :] * Lambda[1, :] + lam_neg[2, :] * Lambda[2, :],
        (gamma - 1) * lam_neg[0, :] * Lambda[0, :] * Lambda[0, :] +
        0.5 * lam_neg[1, :] * Lambda[1, :] * Lambda[1, :] +
        0.5 * lam_neg[2, :] * Lambda[2, :] * Lambda[2, :] + 0.5 * (3 - gamma) /
        (gamma - 1) * (lam_neg[1, :] + lam_neg[2, :]) * c * c
    ])
    return f_pos, f_neg


def h_u(U, delta_x):
    f_pos, f_neg = fvs(U)
    px1 = p_x(f_pos, True, delta_x)
    px2 = p_x(f_neg, False, delta_x)
    px = px1 + px2
    return -px


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


def init(x):
    u = func_u(x)
    rho = func_rho(x)
    p = func_p(x)
    rho_u = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u * u
    U = np.array([rho, rho_u, E])
    return U


def update(x, t):
    num = t / delta_t
    U = init(x)
    rho_init = U[0, :]
    u_init = U[1, :] / rho_init
    p_init = (gamma - 1) * (U[2, :] - 0.5 * rho_init * u_init * u_init)
    U_next = step_time(U, delta_x, delta_t)
    if num > 1:
        for _ in range(int(num - 1)):
            U_next = step_time(U_next, delta_x, delta_t)
    else:
        pass
    rho = U_next[0, :]
    u = U_next[1, :] / rho
    p = (gamma - 1) * (U_next[2, :] - 0.5 * rho * u * u)
    return rho_init, u_init, p_init, rho, u, p


if __name__ == '__main__':
    x = np.linspace(0, 1, 101, endpoint=True)
    t = 0.14
    rho_init, u_init, p_init, rho, u, p = update(x, t)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(x, rho_init, label=r'$\rho_{init}$')
    ax.plot(x, rho, label=r'$\rho_{{t={time}}}$'.format(time=t))
    # ax.plot(x, u_init, label=r'$u_{init}$')
    ax.plot(x, u, label=r'$u_{{t={time}}}$'.format(time=t))
    # ax.plot(x, p_init, label=r'$p_{init}$')
    ax.plot(x, p, label=r'$p_{{t={time}}}$'.format(time=t))
    ax.legend()
    # plt.show()
    fig.savefig('../figures/sod_cal.png', dpi=500)
    plt.show()
