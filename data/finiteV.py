import numpy as np
from matplotlib import pyplot as plt
import re
import time
class FiniteV(object):
    def __init__(self, filepath, T_inf=226.5, Ma_inf=6, 
                 Re_inf=10000, rho_inf=1, r=1, R0=287, gamma=1.4):
        self.k = 0.026 # 取空气的导热系数为0.026
        self.T_inf = T_inf
        self.Ma_inf = Ma_inf
        self.Re_inf = Re_inf
        self.rho_inf = rho_inf
        self.r = r
        self.R0 = R0
        self.gamma = gamma
        self.__parameter_init() # 无穷远处边界条件
        self.__read_net(filepath) # 读取网格
        self.__get_cell_area() # 计算网格面积
        self.__node_data_init() # 设置虚拟网格点
        self.__get_cell_center() # 网格中心坐标
        self.__get_part_xy_xi_ita()
        return

    def __parameter_init(self):
        self.T_wall = self.T_inf * 4.2
        c_inf = np.sqrt(self.gamma * self.R0 * self.T_inf)
        self.u_inf = self.Ma_inf * c_inf
        self.mu_inf =  self.rho_inf * self.r * self.u_inf / self.Re_inf
        self.v_inf = 0
        return
    
    # 读取网格数据
    def __read_net(self, filepath):
        with open(filepath) as file:
            for i,line in enumerate(file):
                if i == 0:
                    print("Reading the data!")
                if i == 1:
                    str_line = re.split('\s+', line)
                    self.m, self.n = int(str_line[3]), int(str_line[5])
                    self.x = np.zeros(shape=(self.m, self.n))
                    self.y = np.zeros(shape=(self.m, self.n))
                if i > 1:
                    loc_0, loc_1 = divmod(i - 2, self.m) # 获取存储位置
                    _, str1, str2, _ = re.split('\s+', line)
                    self.x[self.m - loc_1 - 1, self.n - loc_0 - 1] = float(str1)
                    self.y[self.m - loc_1 - 1, self.n - loc_0 - 1] = float(str2)
            print("Reading complete!")
        return
    
    # 计算每个单元格的面积
    def __get_cell_area(self):
        s1 =(self.x[0:-1, 1:] - self.x[0:-1, 0:-1]) * (self.y[1:, 1:] - self.y[0:-1,0:-1]) - (self.x[1:, 1:]
        - self.x[0:-1, 0:-1]) * (self.y[0:-1, 1:] -self.y[0:-1, 0:-1])
        s2 = (self.x[1:, 0:-1] - self.x[0:-1, 0:-1]) * (self.y[1:, 1:] - self.y[0:-1, 0:-1]) - (self.x[1:, 1:]
         - self.x[0:-1, 0:-1]) * (self.y[1:, 0:-1] - self.y[0:-1, 0:-1]) 
        self.cell_area = 0.5 * np.abs(s1) + 0.5 * np.abs(s2)
        return None
    
    def __node_data_init(self):
        self.node_data = np.zeros(shape=(4, self.m-1, self.n-1))
        self.node_data[0, :, :] = 1
        self.__set_boundary()
        return
    

    def __set_boundary(self):
        self.node_data[0, :, 0] =  2 * self.rho_inf - self.node_data[0, :, 1]
        self.node_data[1, :, 0] = 2 * self.u_inf - self.node_data[1, :, 1]
        self.node_data[2, :, 0] = 2 * self.v_inf - self.node_data[2, :, 1]
        self.node_data[3, :, 0] = 2 * self.T_inf  - self.node_data[3, :, 1]

        self.node_data[0, :, -1] = self.node_data[0, :, -2]
        self.node_data[1, :, -1] = - self.node_data[1, :, -2]
        self.node_data[2, :, -1] = - self.node_data[2, :, -2]
        self.node_data[3, :, -1] = self.T_wall
        self.node_data[3, :, -1] = 0

        self.node_data[:, 0, 1:-1] = self.node_data[:, 1, 1:-1]
        self.node_data[:, -1, 1:-1] = self.node_data[:, -2, 1:-1]
        return
    
    # 由rho， u， v， T计算通量rho,u...等
    def __trans_u2rhou(self, data):
        p = data[0, :, :] * self.R0 * data[3, :, :]
        E = p / (self.gamma - 1) + 0.5 * data[0, :, :] * (data[1, :, :] * data[1, :, :] + data[2, :, :] * data[2, :, :])
        f1 = np.zeros_like(data)
        f2 = np.zeros_like(data)
        f1[0, :, :] = data[0, :, :] * data[1, :, :]
        f1[1, :, :] = data[0, :, :] * data[1, :, :] * data[1, :, :] + p
        f1[2, :, :] = data[0, :, :] * data[1, :, :] * data[2, :, :]
        f1[3, :, :] = data[1, :, :] * (E + p)

        f2[0, :, :] = data[0, :, :] * data[2, :, :]
        f2[1, :, :] = data[0, :, :] * data[1, :, :] * data[2, :, :]
        f2[2, :, :] = data[0, :, :] * data[2, :, :] * data[2, :, :] + p
        f2[3, :, :] = data[2, :, :] * (E + p)
        return f1, f2
    
    # 存储的p,u, v, T --> p, pu, pv, E
    def __trans_T2E(self, data):
        new_data = np.zeros_like(data)
        p = data[0, :, :] * self.R0 * data[3, :, :]
        E = p / (self.gamma - 1) + 0.5 * data[0, :, :] * (data[1, :, :] * data[1, :, :] + data[2, :, :] * data[2, :, :])
        new_data[0, :, :] = data[0, :, :]
        new_data[1, :, :] = data[0, :, :] * data[1, :, :]
        new_data[2, :, :] = data[0, :, :] * data[2, :, :]
        new_data[3, :, :] = E
        return new_data
    
    def __trans_E2T(self, data):
        rho = data[0, :, :]
        u = data[1, :, :]/data[0, :, :]
        v = data[2, :, :]/data[0, :, :]
        p = (data[3, :, :] - 0.5 * rho * (u*u + v*v)) * (self.gamma - 1)
        T = p / rho /self.R0
        new_data = np.zeros_like(data)
        new_data[0, :, :] = rho
        new_data[1, :, :] = u
        new_data[2, :, :] = v
        new_data[3, :, :] = T
        return new_data

    # 计算无粘边界通量
    def __cal_edge_flux(self):
        vert_param = 0.5 * (self.node_data[:, 1:-1, 0:-1] + self.node_data[:, 1:-1, 1:])
        horz_param = 0.5 * (self.node_data[:, 0:-1, 1:-1] + self.node_data[:, 1:, 1:-1])

        v_dy = np.abs(self.y[1:-2, 1:-1] - self.y[2:-1, 1:-1])
        v_dx = np.abs(self.x[1:-2, 1:-1] - self.x[2:-1, 1:-1])
        h_dy = np.abs(self.y[1:-1, 1:-2] - self.y[1:-1, 2:-1])
        h_dx = np.abs(self.x[1:-1, 1:-2] - self.x[1:-1, 2:-1])

        vert_f1, vert_f2 = self.__trans_u2rhou(vert_param)
        horz_f1, horz_f2 = self.__trans_u2rhou(horz_param)

        vert_flux = vert_f1 * v_dy + vert_f2 * v_dx
        horz_flux = horz_f1 * h_dy + horz_f2 * h_dx

        return vert_flux, horz_flux
    
    # 计算单元格无粘通量
    def __cal_inv_flux(self, vert_flux, horz_flux):
        v_flux = vert_flux[:, :, 1:] - vert_flux[:, :, 0:-1]
        h_flux = horz_flux[:, 0:-1, :] - horz_flux[:, 1:, :]
        inv_flux = h_flux + v_flux
        np.savetxt("./invflux.txt", horz_flux[1, :, :], fmt="%f", delimiter=',')
        return inv_flux

    def __cal_mu(self, T):
        # 计算粘性系数
        C = 110.4 / self.T_inf
        mu = self.mu_inf * np.power(T / self.T_inf, 1.5) * (1 + C) / (T / self.T_inf + C)
        return mu
    
    def __get_cell_center(self):
        self.x_cen = 0.25 * (self.x[0:-1, 0:-1] + self.x[1:, 0:-1] + self.x[0:-1, 1:] + self.x[1:, 1:])
        self.y_cen = 0.25 * (self.y[0:-1, 0:-1] + self.y[1:, 0:-1] + self.y[0:-1, 1:] + self.y[1:, 1:])
        return
    
    # 由x对xi的导数转化为xi对x的导数
    def __cal_xi_ita_x_y(self, x_xi, x_ita, y_xi, y_ita):
        J_ = x_xi * y_ita - y_xi * x_ita
        xi_x = y_ita / J_
        xi_y = - x_ita / J_
        ita_x = - y_xi / J_
        ita_y = x_xi / J_ 
        return xi_x, xi_y, ita_x, ita_y
    
    
    def __cal_edge_part_xi_ita(self, var):
        # 竖直两边的梯度
        vert_var_xi = var[1:-1, 1:] - var[1:-1, 0:-1]
        vert_var_ita = 0.25 * (var[0:-2, 0:-1] + var[0:-2, 1:] - var[2:, 0:-1] -var[2:, 1:])

        # 水平两边的梯度
        horz_var_xi = 0.25 * (var[0:-1, 2:] + var[1:, 2:] -var[0:-1, 0:-2] - var[1:, 0:-2])
        horz_var_ita = var[0:-1, 1:-1] - var[1:, 1:-1]
        return  vert_var_xi, vert_var_ita, horz_var_xi, horz_var_ita
    
    def __get_part_xy_xi_ita(self):
        self.vert_x_xi = 0.25 * (self.x[1:-2, 2:] + self.x[2:-1, 2:] - self.x[1:-2, 0:-2] - self.x[2:-1, 0:-2])
        self.vert_x_ita = self.x[1:-2, 1:-1] - self.x[2:-1, 1:-1]
        self.vert_y_xi = 0.25 * (self.y[1:-2, 2:] + self.y[2:-1, 2:] - self.y[1:-2, 0:-2] - self.y[2:-1, 0:-2])
        self.vert_y_ita = self.y[1:-2, 1:-1] - self.y[2:-1, 1:-1]

        self.horz_x_xi = self.x[1:-1, 2:-1] - self.x[1:-1, 1:-2]
        self.horz_x_ita = 0.25 * (self.x[0:-2, 2:-1] + self.x[0:-2, 1:-2] - self.x[2:, 2:-1] - self.x[2:, 1:-2])
        self.horz_y_xi = self.y[1:-1, 2:-1] - self.y[1:-1, 1:-2]
        self.horz_y_ita = 0.25 * (self.y[0:-2, 2:-1] + self.y[0:-2, 1:-2] - self.y[2:, 2:-1] - self.y[2:, 1:-2])
        return
    
        # 将变量对xi和ita的偏导数转化为对x，y的偏导数
    def __cal_var_x_y(self, var_xi, var_ita, xi_x, ita_x, xi_y, ita_y):
        var_x = var_xi * xi_x + var_ita * ita_x
        var_y = var_xi * xi_y + var_ita * ita_y
        return var_x, var_y

    
    # 计算T, u, v的边导数, 对x和y
    def __edge_part_xi_ita(self):
        u = self.node_data[1, :, :]
        v = self.node_data[2, :, :]
        T = self.node_data[3, :, :]
        vert_u_xi, vert_u_ita, horz_u_xi, horz_u_ita = self.__cal_edge_part_xi_ita(u)
        vert_v_xi, vert_v_ita, horz_v_xi, horz_v_ita = self.__cal_edge_part_xi_ita(v)
        vert_T_xi, vert_T_ita, horz_T_xi, horz_T_ita = self.__cal_edge_part_xi_ita(T)
        vert_x_xi, vert_x_ita, horz_x_xi, horz_x_ita = self.vert_x_xi, self.vert_x_ita, self.horz_x_xi, self.horz_x_ita
        vert_y_xi, vert_y_ita, horz_y_xi, horz_y_ita = self.vert_y_xi, self.vert_y_ita, self.horz_y_xi, self.horz_y_ita
        vert_xi_x, vert_xi_y, vert_ita_x, vert_ita_y = self.__cal_xi_ita_x_y(vert_x_xi, vert_x_ita, vert_y_xi, vert_y_ita)
        horz_xi_x, horz_xi_y, horz_ita_x, horz_ita_y = self.__cal_xi_ita_x_y(horz_x_xi, horz_x_ita, horz_y_xi, horz_y_ita)
        
        vert_u_x, vert_u_y = self.__cal_var_x_y(vert_u_xi, vert_u_ita, vert_xi_x, vert_ita_x, vert_xi_y, vert_ita_y)
        vert_v_x, vert_v_y = self.__cal_var_x_y(vert_v_xi, vert_v_ita, vert_xi_x, vert_ita_x, vert_xi_y, vert_ita_y)
        vert_T_x, vert_T_y = self.__cal_var_x_y(vert_T_xi, vert_T_ita, vert_xi_x, vert_ita_x, vert_xi_y, vert_ita_y)

        horz_u_x, horz_u_y = self.__cal_var_x_y(horz_u_xi, horz_u_ita, horz_xi_x, horz_ita_x, horz_xi_y, horz_ita_y)
        horz_v_x, horz_v_y = self.__cal_var_x_y(horz_v_xi, horz_v_ita, horz_xi_x, horz_ita_x, horz_xi_y, horz_ita_y)
        horz_T_x, horz_T_y = self.__cal_var_x_y(horz_T_xi, horz_T_ita, horz_xi_x, horz_ita_x, horz_xi_y, horz_ita_y)
        vert_list = [vert_u_x, vert_u_y, vert_v_x, vert_v_y, vert_T_x, vert_T_y]
        horz_list = [horz_u_x, horz_u_y, horz_v_x, horz_v_y, horz_T_x, horz_T_y]
        return vert_list, horz_list
    
    def __get_mu(self):
        T = self.node_data[3, :, :]
        vert_T = 0.5 * (T[1:-1, 0:-1] + T[1:-1, 1:])
        horz_T = 0.5 * (T[0:-1, 1:-1] + T[1:, 1:-1])
        vert_mu = self.__cal_mu(vert_T)
        horz_mu = self.__cal_mu(horz_T)
        return vert_mu, horz_mu
    
    # 计算粘性项相关参数
    def __cal_v_parameter(self):
        u = self.node_data[1, :, :]
        v = self.node_data[2, :, :]
        vert_list, horz_list = self.__edge_part_xi_ita()
        vert_u_x, vert_u_y, vert_v_x, vert_v_y, vert_T_x, vert_T_y = vert_list
        horz_u_x, horz_u_y, horz_v_x, horz_v_y, horz_T_x, horz_T_y = horz_list
        vert_mu, horz_mu = self.__get_mu()
        vert_tau_xx = vert_mu * (4 * vert_u_x - 2 * vert_v_y)/3
        vert_tau_xy = vert_mu * (vert_u_y + vert_v_x)
        vert_tau_yy = vert_mu * (4 * vert_v_y - 2 * vert_u_x)/3
        vert_tau_yx = vert_mu * (vert_u_y + vert_v_x)

        horz_tau_yy = horz_mu * (4 * horz_v_y - 2 * horz_u_x)/3
        horz_tau_yx = horz_mu * (horz_u_y + horz_v_x)
        horz_tau_xy = horz_mu * (horz_u_y + horz_v_x)
        horz_tau_xx = horz_mu * (4 * horz_u_x - 2 * horz_v_y)/3

        vert_u = 0.5 * (u[1:-1, 0:-1] + u[1:-1, 1:])
        vert_v = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])
        horz_u = 0.5 * (u[0:-1, 1:-1] + u[0:-1, 1:-1])
        horz_v = 0.5 * (v[0:-1, 1:-1] + v[0:-1, 1:-1])
        vert_e_x = self.k * vert_T_x + vert_u * vert_tau_xx + vert_v * vert_tau_xy
        vert_e_y = self.k * vert_T_y + vert_u * vert_tau_yx + vert_v * vert_tau_yy
        horz_e_x = self.k * horz_T_x + horz_u * horz_tau_xx + horz_v * horz_tau_xy
        horz_e_y = self.k * horz_T_y + horz_u * horz_tau_yx + horz_v * horz_tau_yy

        m, n = vert_u.shape
        vert_g1 = np.zeros(shape=(4, m, n))
        vert_g2 = np.zeros(shape=(4, m, n))
        vert_g1[0, :, :] = 0
        vert_g1[1, :, :] = vert_tau_xx
        vert_g1[2, :, :] = vert_tau_xy
        vert_g1[3, :, :] = vert_e_x
        vert_g2[0, :, :] = 0
        vert_g2[1, :, :] = vert_tau_yx
        vert_g2[2, :, :] = vert_tau_yy
        vert_g2[3, :, :] = vert_e_y
        m,n = horz_u.shape
        horz_g1 = np.zeros(shape=(4, m, n))
        horz_g2 = np.zeros(shape=(4, m, n))
        horz_g1[0, :, :] = 0
        horz_g1[1, :, :] = horz_tau_xx
        horz_g1[2, :, :] = horz_tau_xy
        horz_g1[3, :, :] = horz_e_x
        horz_g2[0, :, :] = 0
        horz_g2[1, :, :] = horz_tau_yx
        horz_g2[2, :, :] = horz_tau_yy
        horz_g2[3, :, :] = horz_e_y
        return vert_g1, vert_g2, horz_g1, horz_g2

    # 计算粘性通量
    def __cal_v_flux(self):
        vert_g1, vert_g2, horz_g1, horz_g2 = self.__cal_v_parameter()
        v_dy = np.abs(self.y[1:-2, 1:-1] - self.y[2:-1, 1:-1])
        v_dx = np.abs(self.x[1:-2, 1:-1] - self.x[2:-1, 1:-1])
        h_dy = np.abs(self.y[1:-1, 1:-2] - self.y[1:-1, 2:-1])
        h_dx = np.abs(self.x[1:-1, 1:-2] - self.x[1:-1, 2:-1])
        vert_flux = vert_g1 * v_dy + vert_g2 * v_dx
        horz_flux = horz_g1 * h_dy + horz_g2 * h_dx
        v_flux = vert_flux[:, :, 1:] - vert_flux[:, :, 0:-1] + horz_flux[:, 0:-1, :] - horz_flux[:, 1:, :]
        return v_flux

    
    def __converged(self, node_data, epsion):
        error = node_data[:, 1:-1, 1:-1] - self.node_data[:, 1:-1, 1:-1]
        error_value = np.abs(error).max()
        print("误差为：{}".format(error_value))
        if error_value < epsion:
            return True
        return False

    
    def __iterator(self, dt, epsion):
        curr_U = self.__trans_T2E(self.node_data)
        next_U = np.zeros_like(self.node_data)
        node_data = np.zeros_like(self.node_data)

        vert_flux, horz_flux = self.__cal_edge_flux()
        inv_flux = self.__cal_inv_flux(vert_flux, horz_flux)
        v_flux = self.__cal_v_flux()
    
        next_U[:, 1:-1, 1:-1] = (v_flux-inv_flux) * dt / self.cell_area[1:-1, 1:-1] + curr_U[:, 1:-1, 1:-1]
        data = self.__trans_E2T(next_U[:, 1:-1, 1:-1])
        node_data[:, 1:-1, 1:-1] = data
        is_converged = self.__converged(node_data, epsion)
        return node_data, is_converged
    
    def calc(self, dt=1e-10, epsion=1e-9, max_iter=2):
        for i in range(int(max_iter)):
            # time.sleep(5)
            print('第{}次迭代中...'.format(i+1))
            node_data, is_converged = self.__iterator(dt, epsion)
            self.node_data = node_data
            print(self.node_data[3, :, :].min())
            self.__set_boundary()
            np.savetxt("./rho.txt", self.node_data[0, :, :], fmt="%f", delimiter=',')
            np.savetxt("./u.txt", self.node_data[1, :, :], fmt="%f", delimiter=',')
            np.savetxt("./T.txt", self.node_data[3, :, :],  delimiter=',')

            if is_converged:
                print("The calculate has converged!")
                break
        else:
            print("The result is not converged!")
        return





if __name__ == '__main__':
    finite = FiniteV('/home/xfw/cfd/data/mesh2d.dat')
    finite.calc()
    # print(finite.x[0:2, 0:2])
    # print(finite.y[0:2, 0:2])











