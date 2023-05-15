from matplotlib import pyplot as plt
import numpy as np
from plotstyles import tools

tools.set_mpl_rcParams()

# def func(x):
#     return 0.1781 * np.sqrt(x) - 0.0756 * x - 0.2122 * np.power(
#         x, 2) + 0.1705 * np.power(x, 3) - 0.0609 * np.power(x, 4)

# x = np.linspace(0, 1, 1000, endpoint=True)
# y = func(x)

# fig = plt.figure(figsize=tools.cm2inch([12, 9]))
# ax = fig.add_subplot(111)
# ax.plot(x, y, c='y')
# ax.plot(x, -y, c='y')
# ax.vlines(4, ymin=-2, ymax=2, colors='k')
# ax.hlines([-2, 2], xmin=0, xmax=4, colors='k')
# field_x = np.linspace(-2, 0, endpoint=1000)
# field_y = np.sqrt(4 - field_x**2)
# ax.plot(field_x, field_y, c='k')
# ax.plot(field_x, -field_y, c='k')
# ax.hlines(0, xmin=1, xmax=4, colors='k')
# ax.set_aspect(1)
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# fig.savefig('../figures/net.pdf')


def half_wing(x):
    funcVal = 0.1781 * np.sqrt(
        x) - 0.0756 * x - 0.2122 * x**2 + 0.1705 * x**3 - 0.0609 * x**4
    return funcVal


# 构造物理空间与计算空间边界映射关系
def build_bdy_maps():
    p2c_xi = [
        ((1.0, 0.0), (0, 15)), ((0.9, half_wing(0.9)), (1, 15)),
        ((0.8, half_wing(0.8)), (2, 15)), ((0.7, half_wing(0.7)), (3, 15)),
        ((0.6, half_wing(0.6)), (4, 15)), ((0.5, half_wing(0.5)), (5, 15)),
        ((0.4, half_wing(0.4)), (6, 15)), ((0.3, half_wing(0.3)), (7, 15)),
        ((0.2, half_wing(0.2)), (8, 15)), ((0.1, half_wing(0.1)), (9, 15)),
        ((0.0, half_wing(0.0)), (10, 15)), ((0.1, -half_wing(0.1)), (11, 15)),
        ((0.2, -half_wing(0.2)), (12, 15)), ((0.3, -half_wing(0.3)), (13, 15)),
        ((0.4, -half_wing(0.4)), (14, 15)), ((0.5, -half_wing(0.5)), (15, 15)),
        ((0.6, -half_wing(0.6)), (16, 15)), ((0.7, -half_wing(0.7)), (17, 15)),
        ((0.8, -half_wing(0.8)), (18, 15)), ((0.9, -half_wing(0.9)), (19, 15)),
        ((1.0, 0.0), (20, 15)), ((4.0, 0.0), (0, 0)), ((4.0, 1.0), (1, 0)),
        ((4.0, 2.0), (2, 0)), ((3.0, 2.0), (3, 0)), ((2.0, 2.0), (4, 0)),
        ((1.0, 2.0), (5, 0)), ((0.0, 2.0), (6, 0)), ((-1.0, 2.0), (7, 0)),
        ((-2.0, 2.0), (8, 0)), ((-2.0, 1.0), (9, 0)), ((-2.0, 0.0), (10, 0)),
        ((-2.0, -1.0), (11, 0)), ((-2.0, -2.0), (12, 0)),
        ((-1.0, -2.0), (13, 0)), ((0.0, -2.0), (14, 0)),
        ((1.0, -2.0), (15, 0)), ((2.0, -2.0), (16, 0)), ((3.0, -2.0), (17, 0)),
        ((4.0, -2.0), (18, 0)), ((4.0, -1.0), (19, 0)), ((4.0, 0.0), (20, 0))
    ]
    p2c_eta = [
        ((4.0, 0.0), (0, 0)), ((3.8, 0.0), (0, 1)), ((3.6, 0.0), (0, 2)),
        ((3.4, 0.0), (0, 3)), ((3.2, 0.0), (0, 4)), ((3.0, 0.0), (0, 5)),
        ((2.8, 0.0), (0, 6)), ((2.6, 0.0), (0, 7)), ((2.4, 0.0), (0, 8)),
        ((2.2, 0.0), (0, 9)), ((2.0, 0.0), (0, 10)), ((1.8, 0.0), (0, 11)),
        ((1.6, 0.0), (0, 12)), ((1.4, 0.0), (0, 13)), ((1.2, 0.0), (0, 14)),
        ((1.0, 0.0), (0, 15)), ((4.0, 0.0), (20, 0)), ((3.8, 0.0), (20, 1)),
        ((3.6, 0.0), (20, 2)), ((3.4, 0.0), (20, 3)), ((3.2, 0.0), (20, 4)),
        ((3.0, 0.0), (20, 5)), ((2.8, 0.0), (20, 6)), ((2.6, 0.0), (20, 7)),
        ((2.4, 0.0), (20, 8)), ((2.2, 0.0), (20, 9)), ((2.0, 0.0), (20, 10)),
        ((1.8, 0.0), (20, 11)), ((1.6, 0.0), (20, 12)), ((1.4, 0.0), (20, 13)),
        ((1.2, 0.0), (20, 14)), ((1.0, 0.0), (20, 15))
    ]
    return p2c_xi, p2c_eta


class EllipticGrid(object):

    def __init__(self, p2c_xi, p2c_eta):
        self.__p2c_xi = p2c_xi  # 物理空间与ξ轴的边界映射关系
        self.__p2c_eta = p2c_eta  # 物理空间与eta轴的边界映射关系

        self.__n_xi, self.__n_eta = self.__get_n()  # 计算空间子区间网格划分数
        self.__ht = 0.1

    def get_solu(self, max_iter=1000000, epsilon=1.e-9):
        '''
        数值求解
        max_iter:最大迭代次数
        epsilon：收敛判据
        '''
        X0, Y0 = self.__get_init()

        for _ in range(max_iter):
            Xx = self.__calc_Ux(X0)
            Xe = self.__calc_Ue(X0)
            Xxx = self.__calc_Uxx(X0)
            Xee = self.__calc_Uee(X0)
            Xxe = self.__calc_Uxe(X0)
            Yx = self.__calc_Ux(Y0)
            Ye = self.__calc_Ue(Y0)
            Yxx = self.__calc_Uxx(Y0)
            Yee = self.__calc_Uee(Y0)
            Yxe = self.__calc_Uxe(Y0)

            alpha = self.__calc_alpha(Xe, Ye)
            beta = self.__calc_beta(Xx, Xe, Yx, Ye)
            gamma = self.__calc_gamma(Xx, Yx)

            Kx = alpha * Xxx - 2 * beta * Xxe + gamma * Xee
            Ky = alpha * Yxx - 2 * beta * Yxe + gamma * Yee

            Xk = self.__step_U(X0, Kx)
            Yk = self.__step_U(Y0, Ky)
            print(Xk)

            if self.__converged(Xk - X0, Yk - Y0, epsilon):
                break

        else:
            raise Exception(
                ">>> Not converged after {} iterations! <<<".format(max_iter))
        return Xk, Yk

    def __converged(self, deltaX, deltaY, epsilon):
        norm_val1 = np.linalg.norm(deltaX, np.inf)
        norm_val2 = np.linalg.norm(deltaY, np.inf)
        if norm_val1 < epsilon and norm_val2 < epsilon:
            return True
        return False

    def __calc_Ux(self, U):
        Ux = np.zeros_like(U)
        Ux[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / 2
        return Ux

    def __calc_Ue(self, U):
        Ue = np.zeros_like(U)
        Ue[1:-1, :] = (U[2:, :] - U[:-2, :]) / 2
        return Ue

    def __calc_Uxx(self, U):
        Uxx = np.zeros_like(U)
        Uxx[:, 1:-1] = U[:, 2:] + U[:, :-2] - 2 * U[:, 1:-1]
        return Uxx

    def __calc_Uee(self, U):
        Uee = np.zeros_like(U)
        Uee[1:-1, :] = U[2:, :] + U[:-2, :] - 2 * U[1:-1, :]
        return Uee

    def __calc_Uxe(self, U):
        Uxe = np.zeros_like(U)
        Uxe[1:-1,
            1:-1] = (U[2:, 2:] + U[:-2, :-2] - U[:-2, 2:] - U[2:, :-2]) / 4
        return Uxe

    def __calc_alpha(self, Xe, Ye):
        return Xe**2 + Ye**2

    def __calc_beta(self, Xx, Xe, Yx, Ye):
        return -Xx * Xe - Yx * Ye

    def __calc_gamma(self, Xx, Yx):
        return Xx**2 + Yx**2

    def __step_U(self, U, K):
        Uk = np.copy(U)
        Uk[1:-1, 1:-1] = U[1:-1, 1:-1] + K[1:-1, 1:-1] * self.__ht
        return Uk

    def __get_n(self):
        arr_xi = np.array(list(item[1] for item in self.__p2c_xi))
        n_xi = np.max(arr_xi[:, 0])
        n_eta = np.max(arr_xi[:, 1])
        return n_xi, n_eta

    def __get_init(self):
        X = np.zeros(shape=(self.__n_eta + 1, self.__n_xi + 1))
        Y = np.zeros(shape=(self.__n_eta + 1, self.__n_xi + 1))
        for XY, XE in self.__p2c_xi:
            X[XE[1], XE[0]] = XY[0]
            Y[XE[1], XE[0]] = XY[1]
        for XY, XE in self.__p2c_eta:
            X[XE[1], XE[0]] = XY[0]
            Y[XE[1], XE[0]] = XY[1]
        return X, Y


class EGPlot(object):

    @staticmethod
    def plot_fig(egObj):
        maxIter = 1000000
        epsilon = 1.e-9

        X, Y = egObj.get_solu(maxIter, epsilon)

        fig = plt.figure(figsize=(6, 4))
        ax1 = plt.subplot()

        ax1.plot(X[:, 0], Y[:, 0], c="red", lw=1, label="mapping to $\\eta$")
        ax1.plot(X[:, -1], Y[:, -1], c="red", lw=1)
        n_eta, n_xi = X.shape
        for colIdx in range(1, n_xi - 1):
            tmpX = X[:, colIdx]
            tmpY = Y[:, colIdx]
            ax1.plot(tmpX, tmpY, "k-", lw=1)

        ax1.plot(X[0, :], Y[0, :], c="green", lw=1, label="mapping to $\\xi$")
        ax1.plot(X[-1, :], Y[-1, :], c="green", lw=1)
        for rowIdx in range(1, n_eta - 1):
            tmpX = X[rowIdx, :]
            tmpY = Y[rowIdx, :]
            ax1.plot(tmpX, tmpY, "k-", lw=1)
        ax1.legend()

        fig.tight_layout()
        fig.savefig("plot_fig.png", dpi=100)


if __name__ == "__main__":
    p2c_xi, p2c_eta = build_bdy_maps()
    egObj = EllipticGrid(p2c_xi, p2c_eta)
    EGPlot.plot_fig(egObj)
