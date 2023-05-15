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
def calc_subtem_coef(order, cond=True):
    suborder = int((order + 1) / 2)
    q = np.zeros(shape=(suborder, order + 2))
    if cond:
        for i in range(suborder):
            r = suborder - i
            _, _, matrix_b = calc(suborder + 1, r, 1)
            q[i, i + 1:i + 1 + suborder] = matrix_b.flatten()
    else:
        for i in range(suborder):
            r = suborder - i - 1
            _, _, matrix_b = calc(suborder + 1, r, 1)
            q[i, i + 2:i + 2 + suborder] = matrix_b.flatten()
    return q


# f_order = c0*q0 + c1*q1 + ... + cr-1 * qr-1
# return [c0, c1, ..., cr-1]
def get_smooth_coef(order, q, cond=True):
    m = order + 1
    k = int((order + 1) / 2)
    if cond:
        _, _, smooth_mat_b = calc(m, k, 1)
        smooth_mat_b_all = np.zeros(order + 2)
        smooth_mat_b_all[1:-1] = smooth_mat_b.flatten()
        c = smooth_mat_b_all.reshape(1, -1).dot(np.linalg.pinv(q))
    else:
        _, _, smooth_mat_b = calc(m, k - 1, 1)
        smooth_mat_b_all = np.zeros(order + 2)
        smooth_mat_b_all[2::] = smooth_mat_b.flatten()
        c = smooth_mat_b_all.reshape(1, -1).dot(np.linalg.pinv(q))
    return c[0]


# only for weno 5
# u = [uj-3, ..., uj+2]
def calc_smooth_ind(u, cond=True):
    if cond:
        is0 = 13 / 12 * np.power(
            (u[:, 1] - 2 * u[:, 2] + u[:, 3]), 2) + 0.25 * np.power(
                (u[:, 1] - 4 * u[:, 2] + 3 * u[:, 3]), 2)
        is1 = 13 / 12 * np.power(
            (u[:, 2] - 2 * u[:, 3] + u[:, 4]), 2) + 0.25 * np.power(
                (u[:, 2] - u[:, 4]), 2)
        is2 = 13 / 12 * np.power(
            (u[:, 3] - 2 * u[:, 4] + u[:, 5]), 2) + 0.25 * np.power(
                (3 * u[:, 3] - 4 * u[:, 4] + u[:, 5]), 2)
    else:
        is0 = 13 / 12 * np.power(
            (u[:, 5] - 2 * u[:, 4] + u[:, 3]), 2) + 0.25 * np.power(
                (u[:, 5] - 4 * u[:, 4] + 3 * u[:, 3]), 2)
        is1 = 13 / 12 * np.power(
            (u[:, 4] - 2 * u[:, 3] + u[:, 2]), 2) + 0.25 * np.power(
                (u[:, 4] - u[:, 2]), 2)
        is2 = 13 / 12 * np.power(
            (u[:, 3] - 2 * u[:, 2] + u[:, 1]), 2) + 0.25 * np.power(
                (3 * u[:, 3] - 4 * u[:, 2] + u[:, 1]), 2)
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


def h_coef(w_all, q):
    w0, w1, w2 = w_all
    param1_coef = w0[0] * q[0, :] + w1[0] * q[1, :] + w2[0] * q[2, :]
    param2_coef = w0[1] * q[0, :] + w1[1] * q[1, :] + w2[1] * q[2, :]
    param3_coef = w0[2] * q[0, :] + w1[2] * q[1, :] + w2[2] * q[2, :]
    param_coef = np.array([param1_coef, param2_coef, param3_coef])
    return param_coef


# 流通矢量分解
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


def p_x(U, delta_x, q_pos, q_neg, c_pos, c_neg):
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
        u = exten_U[:, i:i + 7]
        fu_pos = exten_f_pos[:, i:i + 7]
        fu_neg = exten_f_neg[:, i:i + 7]
        is_all_pos = calc_smooth_ind(u, True)
        is_all_neg = calc_smooth_ind(u, False)
        w_all_pos = cal_omega(c_pos, is_all_pos)
        w_all_neg = cal_omega(c_neg, is_all_neg)
        coef_p = h_coef(w_all_pos, q_pos)
        coef_n = h_coef(w_all_neg, q_neg)
        # 左移一位计算fj-0.5
        coef_p_ = np.roll(coef_p, -1)
        coef_n_ = np.roll(coef_n, -1)
        px_pos = np.sum((coef_p - coef_p_) * fu_pos, axis=1) / delta_x
        px_neg = np.sum((coef_n - coef_n_) * fu_neg, axis=1) / delta_x
        # delta_f = (coef_p * fu_pos).sum(axis=1) + (coef_n * fu_neg).sum(
        #     axis=1) - (coef_p_ * fu_pos).sum(axis=1) - (coef_n_ *
        #                                                 fu_neg).sum(axis=1)
        px[:, i] = px_pos + px_neg
    return px


def h_u(U, delta_x, q_pos, q_neg, c_pos, c_neg):
    return -p_x(U, delta_x, q_pos, q_neg, c_pos, c_neg)


def step_time(U, delta_x, delta_t, q_pos, q_neg, c_pos, c_neg):
    U1 = U + delta_t * h_u(U, delta_x, q_pos, q_neg, c_pos, c_neg)
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * delta_t * h_u(U1, delta_x, q_pos, q_neg,
                                                     c_pos, c_neg)
    U_next = 1 / 3 * U + 2 / 3 * U2 + 2 / 3 * delta_t * h_u(
        U2, delta_x, q_pos, q_neg, c_pos, c_neg)
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
    q_pos = calc_subtem_coef(5, True)
    q_neg = calc_subtem_coef(5, False)
    c_pos = get_smooth_coef(5, q_pos, True)
    c_neg = get_smooth_coef(5, q_neg, False)
    u = func_u(x)
    rho = func_rho(x)
    p = func_p(x)
    rho_u = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u * u
    U = np.array([rho, rho_u, E])
    return U, q_pos, q_neg, c_pos, c_neg


# --------------------------------------------


def update(x, t):
    num = t / delta_t
    U, q_pos, q_neg, c_pos, c_neg = init(x)
    rho_init = U[0, :]
    u_init = U[1, :] / rho_init
    p_init = (gamma - 1) * (U[2, :] - 0.5 * rho_init * u_init * u_init)
    U_next = step_time(U, delta_x, delta_t, q_pos, q_neg, c_pos, c_neg)
    if num > 1:
        for _ in range(int(num - 1)):
            U_next = step_time(U_next, delta_x, delta_t, q_pos, q_neg, c_pos,
                               c_neg)
    else:
        pass
    rho = U_next[0, :]
    u = U_next[1, :] / rho
    p = (gamma - 1) * (U_next[2, :] - 0.5 * rho * u * u)
    return rho_init, u_init, p_init, rho, u, p


def main():
    x = np.linspace(0, 1, 101, endpoint=True)
    t = 0.002
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
