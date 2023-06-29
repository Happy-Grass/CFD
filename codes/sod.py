import numpy as np
import matplotlib.pyplot as plt
import math

gamma = 1.4


def f(p_star, p_i, rho_i):
    if p_i == 0:
        print("Please ensure that p_i is not 0!")
        return None
    if rho_i == 0:
        print("Please ensure that rho_i is not 0!")
        return None
    c_i = math.sqrt(gamma * p_i / rho_i)
    if p_star > p_i:
        result = (p_star - p_i) / (rho_i * c_i *
                                   math.sqrt((gamma + 1) * p_star /
                                             (gamma * 2 * p_i) +
                                             (gamma - 1) / 2 / gamma))
    else:
        result = (2 * c_i / (gamma - 1) *
                  (math.pow(p_star / p_i, (gamma - 1) / 2 / gamma) - 1))
    return result


# 弦截法
def solver(func, ca=0.00000000001, max_iterations=100):
    x0, x1 = 0, 1
    k = 0
    while abs(func(x1)) > ca and k < max_iterations:
        x_next = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        x0 = x1
        x1 = x_next
        k = k + 1
    return x1


def F(p_star, p_1, rho_1, p_2, rho_2, u1, u2):
    return f(p_star, p_1, rho_1) + f(p_star, p_2, rho_2) - u1 + u2


p1, rho1, p2, rho2, u1, u2 = 1, 1, 0.1, 0.125, 0, 0
p_star = solver(lambda x: F(x, p1, rho1, p2, rho2, u1, u2))
u_star = u1 - f(p_star, p1, rho1)
rho_star_l = math.pow(p_star / p1 * math.pow(rho1, gamma), 1 / gamma)
z2 = u2 + (p2 - p_star) / rho2 / (u2 - u_star)
rho_star_r = rho2 * (u2 - z2) / (u_star - z2)


def cal_c(rho, p):
    c = math.sqrt(gamma * p / rho)
    return c


c1 = cal_c(rho1, p1)
c2 = cal_c(rho2, p2)
c_star_l = cal_c(rho_star_l, p_star)
c_star_r = cal_c(rho_star_r, p_star)


def cal_area2(x, t, c1):
    # 计算膨胀波区 x为array
    c = (gakma - 1) / (gamma + 1) * (u1 - (x - 0.5) / t) + 2 * c1 / (gamma + 1)
    u = c + (x - 0.5) / t
    p = p1 * np.power(c / c1, 2 * gamma / (gamma - 1))
    rho = gamma * p / c / c
    return c, u, p, rho


def get_range(u1, c1, u_star, c_star_l, z2, t):
    x1 = (u1 - c1) * t + 0.5
    x2 = (u_star - c_star_l) * t + 0.5
    x3 = u_star * t + 0.5
    x4 = z2 * t + 0.5
    return x1, x2, x3, x4


def cal_value_all(x, t):
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    x1, x2, x3, x4 = get_range(u1, c1, u_star, c_star_l, z2, t)

    # area 1
    rho[x < x1] = rho1
    u[x < x1] = u1
    p[x < x1] = p1

    # area 2
    _, temp_u, temp_p, temp_rho = cal_area2(x[(x >= x1) & (x < x2)], t, c1)
    rho[(x >= x1) & (x < x2)] = temp_rho
    u[(x >= x1) & (x < x2)] = temp_u
    p[(x >= x1) & (x < x2)] = temp_p

    # area 3
    rho[(x >= x2) & (x < x3)] = rho_star_l
    u[(x >= x2) & (x < x3)] = u_star
    p[(x >= x2) & (x < x3)] = p_star

    # area 4
    rho[(x >= x3) & (x < x4)] = rho_star_r
    u[(x >= x3) & (x < x4)] = u_star
    p[(x >= x3) & (x < x4)] = p_star

    # area 5
    rho[x >= x4] = rho2
    u[x >= x4] = u2
    p[x >= x4] = p2
    return rho, u, p


if __name__ == '__main__':
    x = np.linspace(0, 1, 101, endpoint=True)
    rho, u, p = cal_value_all(x, 0.14)
    plt.plot(x, rho, label=r'$\rho$')
    plt.plot(x, p, label='$p$')
    plt.plot(x, u, label='$u$')
    plt.text(0.1, 0.9, 't=0.14')
    plt.legend()
    plt.savefig('../figures/sod_accurate.png', dpi=500)
