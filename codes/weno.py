from coeff_schemes import calc
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
delta_x = 0.01
delta_t = 0.001


# fj-r, ...., fj, ..., fj+r-1
# return a matrix which save as the format of
# [[q0],
#  [q1],
#  [q2],
#  ...
#  [qr-1]]
# 计算odder阶weno格式子模版守恒型通量系数
def calc_subtem_coef(order):
    suborder = int((order + 1) / 2)
    q = np.zeros(shape=(suborder, order + 1))
    for i in range(suborder):
        r = suborder - i
        _, _, matrix_b = calc(suborder + 1, r, 1)
        q[i, i + 1:i + 1 + suborder] = matrix_b.flatten()
    return q


# f_order = c0*q0 + c1*q1 + ... + cr-1 * qr-1
# return [c0, c1, ..., cr-1]
def get_smooth_coef(order, q):
    m = order + 1
    k = int((order + 1) / 2)
    _, _, smooth_mat_b = calc(m, k, 1)
    smooth_mat_b_all = np.zeros(order + 1)
    smooth_mat_b_all[1:] = smooth_mat_b.flatten()
    c = smooth_mat_b_all.reshape(1, -1).dot(np.linalg.pinv(q))
    return c[0]


# only for weno 5
# u = [uj-3, ..., uj+2]
def calc_smooth_ind(u):
    is0 = 13 / 12 * np.power(
        (u[:, 1] - 2 * u[:, 2] + u[:, 3]), 2) + 0.25 * np.power(
            (u[:, 1] - 4 * u[:, 2] + 3 * u[:, 3]), 2)
    is1 = 13 / 12 * np.power(
        (u[:, 2] - 2 * u[:, 3] + u[:, 4]), 2) + 0.25 * np.power(
            (u[:, 2] - u[:, 4]), 2)
    is2 = 13 / 12 * np.power(
        (u[:, 3] - 2 * u[:, 4] + u[:, 5]), 2) + 0.25 * np.power(
            (3 * u[:, 3] - 4 * u[:, 4] + u[:, 5]), 2)
    return (is0, is1, is2)


def cal_omega(c, is_all):
    epsion = 0.0000001
    p = 3
    is0, is1, is2 = is_all
    alpha0 = c[0] / np.power((epsion + is0), p)
    alpha1 = c[1] / np.power((epsion + is1), p)
    alpha2 = c[2] / np.power((epsion + is2), p)
    sum_alpha = alpha0 + alpha1 + alpha2
    return (alpha0 / sum_alpha, alpha1 / sum_alpha, alpha2 / sum_alpha)


# 计算f0.5， a>0的系数
def h_coef(w_all, q):
    w0, w1, w2 = w_all
    param1_coef = w0[0] * q[0, :] + w1[0] * q[1, :] + w2[0] * q[2, :]
    param2_coef = w0[1] * q[0, :] + w1[1] * q[1, :] + w2[1] * q[2, :]
    param3_coef = w0[2] * q[0, :] + w1[2] * q[1, :] + w2[2] * q[2, :]
    param_coef = np.array([param1_coef, param2_coef, param3_coef])
    param_coef_neg = np.flip(param_coef, axis=1)  # a<0时的系数
    param_coef_neg = np.roll(param_coef_neg, 1)
    return param_coef, param_coef_neg


# 流通矢量分解
def fvs(U):
    rho = U[0, :]
    u = U[1, :] / U[0, :]
    p = (gamma - 1) * (U[2, :] - 0.5 * rho * u * u)
    c = np.sqrt(gamma * p / rho)
    lam = np.array([u, u - c, u + c])
    lam_pos = 0.5 * lam + 0.5 * np.abs(lam)
    lam_neg = 0.5 * lam - 0.5 * np.abs(lam)
    f_pos = 0.5 * rho / gamma * np.array([
        2 * (gamma - 1) * lam_pos[0, :] + lam_pos[1, :] + lam_pos[2, :], 2 *
        (gamma - 1) * lam_pos[0, :] * lam[0, :] + lam_pos[1, :] * lam[1, :] +
        lam_pos[2, :] * lam[2, :],
        (gamma - 1) * lam_pos[0, :] * lam[0, :] * lam[0, :] +
        0.5 * lam_pos[1, :] * lam[1, :] * lam[1, :] +
        0.5 * lam_pos[2, :] * lam[2, :] * lam[2, :] + 0.5 * (3 - gamma) /
        (gamma - 1) * (lam_pos[1, :] + lam_pos[2, :]) * c * c
    ])

    f_neg = 0.5 * rho / gamma * np.array([
        2 * (gamma - 1) * lam_neg[0, :] + lam_neg[1, :] + lam_neg[2, :], 2 *
        (gamma - 1) * lam_neg[0, :] * lam[0, :] + lam_neg[1, :] * lam[1, :] +
        lam_neg[2, :] * lam[2, :],
        (gamma - 1) * lam_neg[0, :] * lam[0, :] * lam[0, :] +
        0.5 * lam_neg[1, :] * lam[1, :] * lam[1, :] +
        0.5 * lam_neg[2, :] * lam[2, :] * lam[2, :] + 0.5 * (3 - gamma) /
        (gamma - 1) * (lam_neg[1, :] + lam_neg[2, :]) * c * c
    ])
    return f_pos, f_neg


def p_x(U, delta_x, q, c):
    px = U.copy()
    m, n = U.shape
    f_pos, f_neg = fvs(U)
    exten_f_pos = np.zeros(shape=(m, n + 6))
    exten_U = np.zeros(shape=(m, n + 6))
    exten_f_neg = np.zeros(shape=(m, n + 6))
    exten_f_pos[:, 3:-3] = f_pos
    exten_f_pos[:, 0:3] = f_pos[:, 0:3]
    exten_f_pos[:, -3::] = f_pos[:, -3::]
    exten_f_neg[:, 3:-3] = f_neg
    exten_f_neg[:, 0:3] = f_neg[:, 0:3]
    exten_f_neg[:, -3::] = f_neg[:, -3::]
    exten_U[:, 3:-3] = U
    exten_U[:, 0:3] = U[:, 0:3]
    exten_U[:, -3::] = U[:, -3::]
    for i in range(n):
        u = exten_U[:, i:i + 6]
        fu_pos = exten_f_pos[:, i:i + 6]
        fu_neg = exten_f_neg[:, i:i + 6]
        is_all = calc_smooth_ind(u)
        w_all = cal_omega(c, is_all)
        coef_p, coef_n = h_coef(w_all, q)
        # 左移一位计算fj-0.5
        coef_p_ = np.roll(coef_p, -1)
        coef_n_ = np.roll(coef_n, -1)
        delta_f = (coef_p * fu_pos).sum(axis=1) + (coef_n * fu_neg).sum(
            axis=1) - (coef_p_ * fu_pos).sum(axis=1) - (coef_n_ *
                                                        fu_neg).sum(axis=1)
        px[:, i] = delta_f / delta_x
    print(px)
    return px


def h_u(U, delta_x, q, c):
    return -p_x(U, delta_x, q, c)


def step_time(U, delta_x, delta_t, q, c):
    U1 = U + delta_t * h_u(U, delta_x, q, c)
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * delta_t * h_u(U1, delta_x, q, c)
    U_next = 1 / 3 * U + 2 / 3 * U2 + 2 / 3 * delta_t * h_u(U2, delta_x, q, c)
    return U_next


# 初始化条件 -----------------------------
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
    q = calc_subtem_coef(5)
    cc = get_smooth_coef(5, q)
    u = func_u(x)
    rho = func_rho(x)
    p = func_p(x)
    rho_u = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u * u
    U = np.array([rho, rho_u, E])
    return U, q, cc


# --------------------------------------------


def update(x, t):
    num = t / delta_t
    U, q, cc = init(x)
    rho_init = U[0, :]
    u_init = U[1, :] / rho_init
    p_init = (gamma - 1) * (U[2, :] - 0.5 * rho_init * u_init * u_init)
    U_next = step_time(U, delta_x, delta_t, q, cc)
    if num > 1:
        for _ in range(int(num - 1)):
            U_next = step_time(U_next, delta_x, delta_t, q, cc)
    else:
        pass
    rho = U_next[0, :]
    u = U_next[1, :] / rho
    p = (gamma - 1) * (U_next[2, :] - 0.5 * rho * u * u)
    return rho_init, u_init, p_init, rho, u, p


def main():
    x = np.linspace(0, 1, 101, endpoint=True)
    t = 0.001
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
    plt.show()


main()
