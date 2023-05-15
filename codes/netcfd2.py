# 椭圆型方程网格生成法

import numpy
from matplotlib import pyplot as plt


# 对称翼型的上半部
def half_wing(x):
    funcVal = 0.1781 * numpy.sqrt(
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
        self.__p2c_xi = p2c_xi  # 物理空间与计算空间xi轴边界映射关系
        self.__p2c_eta = p2c_eta  # 物理空间与计算空间eta轴边界映射关系

        self.__n_xi, self.__n_eta = self.__get_n()  # 计算空间子区间划分数
        self.__ht = 0.1

        self.__bdyX, self.__bdyY = self.__get_bdy()

    def get_solu(self, maxIter=1000000, epsilon=1.e-9):
        '''
        数值求解
        maxIter: 最大迭代次数
        epsilon: 收敛判据
        '''
        X0 = self.__get_initX()
        Y0 = self.__get_initY()

        for i in range(maxIter):
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

            Xk = self.__calc_Uk(X0, Kx, self.__ht)
            Yk = self.__calc_Uk(Y0, Ky, self.__ht)
            self.__fill_bdyX(Xk)
            self.__fill_bdyY(Yk)

            # print(i, numpy.linalg.norm(Xk - X0, numpy.inf), numpy.linalg.norm(Yk - Y0, numpy.inf))
            if self.__converged(Xk - X0, Yk - Y0, epsilon):
                break

            X0 = Xk
            Y0 = Yk
        else:
            raise Exception(
                ">>> Not converged after {} iterations! <<<".format(maxIter))

        return Xk, Yk

    def __converged(self, deltaX, deltaY, epsilon):
        normVal1 = numpy.linalg.norm(deltaX, numpy.inf)
        normVal2 = numpy.linalg.norm(deltaY, numpy.inf)
        if normVal1 < epsilon and normVal2 < epsilon:
            return True
        return False

    def __calc_Ux(self, U):
        Ux = numpy.zeros(U.shape)
        Ux[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / 2
        return Ux

    def __calc_Ue(self, U):
        Ue = numpy.zeros(U.shape)
        Ue[1:-1, :] = (U[2:, :] - U[:-2, :]) / 2
        return Ue

    def __calc_Uxx(self, U):
        Uxx = numpy.zeros(U.shape)
        Uxx[:, 1:-1] = U[:, 2:] + U[:, :-2] - 2 * U[:, 1:-1]
        return Uxx

    def __calc_Uee(self, U):
        Uee = numpy.zeros(U.shape)
        Uee[1:-1, :] = U[2:, :] + U[:-2, :] - 2 * U[1:-1, :]
        return Uee

    def __calc_Uxe(self, U):
        Uxe = numpy.zeros(U.shape)
        Uxe[1:-1,
            1:-1] = (U[2:, 2:] + U[:-2, :-2] - U[:-2, 2:] - U[2:, :-2]) / 4
        return Uxe

    def __calc_alpha(self, Xe, Ye):
        alpha = Xe**2 + Ye**2
        return alpha

    def __calc_beta(self, Xx, Xe, Yx, Ye):
        beta = Xx * Xe + Yx * Ye
        return beta

    def __calc_gamma(self, Xx, Yx):
        gamma = Xx**2 + Yx**2
        return gamma

    def __calc_Uk(self, U, K, ht):
        Uk = U + K * ht
        return Uk

    def __get_bdy(self):
        '''
        获取边界条件
        '''
        bdyX = numpy.zeros((self.__n_eta + 1, self.__n_xi + 1))
        bdyY = numpy.zeros((self.__n_eta + 1, self.__n_xi + 1))

        for XY, XE in self.__p2c_xi:
            bdyX[XE[1], XE[0]] = XY[0]
            bdyY[XE[1], XE[0]] = XY[1]

        for XY, XE in self.__p2c_eta:
            bdyX[XE[1], XE[0]] = XY[0]
            bdyY[XE[1], XE[0]] = XY[1]

        return bdyX, bdyY

    def __get_initX(self):
        '''
        获取X之初始条件
        '''
        X0 = numpy.zeros(self.__bdyX.shape)
        self.__fill_bdyX(X0)
        return X0

    def __get_initY(self):
        '''
        获取Y之初始条件
        '''
        Y0 = numpy.zeros(self.__bdyY.shape)
        self.__fill_bdyY(Y0)
        return Y0

    def __fill_bdyX(self, U):
        '''
        填充X之边界条件
        '''
        U[:, 0] = self.__bdyX[:, 0]
        U[:, -1] = self.__bdyX[:, -1]
        U[0, :] = self.__bdyX[0, :]
        U[-1, :] = self.__bdyX[-1, :]

    def __fill_bdyY(self, U):
        '''
        填充Y之边界条件
        '''
        U[:, 0] = self.__bdyY[:, 0]
        U[:, -1] = self.__bdyY[:, -1]
        U[0, :] = self.__bdyY[0, :]
        U[-1, :] = self.__bdyY[-1, :]

    def __get_n(self):
        arr_xi = numpy.array(list(item[1] for item in self.__p2c_xi))
        n_xi = numpy.max(arr_xi[:, 0])
        n_eta = numpy.max(arr_xi[:, 1])
        return n_xi, n_eta


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
        fig.savefig("plot_fig.pdf", dpi=100)


if __name__ == "__main__":
    # p2c_xi, p2c_eta = build_bdy_maps()
    # egObj = EllipticGrid(p2c_xi, p2c_eta)
    print([1, 2, 3, 4, 5])
    # EGPlot.plot_fig(egObj)
