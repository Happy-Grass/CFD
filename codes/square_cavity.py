
import numpy as np 
from matplotlib import pyplot as plt

# 采用交错网格
def init_con(nx, ny, len_x=2, len_y=2):
    dx = len_x / (nx - 1)
    dy = len_y / (ny - 1)
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx))
    return u, v, p, dx, dy


class PrjMethod(object):
    def __init__(self, u, v, p, mu, dx, dy, dt=0.001, max_iter=10000, epsion=1e-10):
        self.u = u
        self.v = v
        self.p = p
        self.mu = mu
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.max_iter= max_iter
        self.epsion= epsion
        self.rho = 1000
    
    def __bnd_v(self):
        self.u[0, :] = 2
        self.u[1:, 0] = 0
        self.u[1:, -1] = 0
        self.v[-1, :] = 0
        self.v[0, :] = 0
        
    # 速度3阶迎风格式(对流项)
    def __partial_var(self, var, is_pos):
        par_var_x = np.zeros_like(var)
        par_var_y = np.zeros_like(var)
        if  is_pos:
            par_var_x[1:-1, 2:-1] = (2 * var[1:-1, 3:] + 3 * var[1:-1, 2:-1] -
                                     6 * var[1:-1, 1:-2] + var[1:-1, 0:-3])/6/self.dx
            par_var_y[2:-1, 1:-1] = (2 * var[3:, 1:-1] + 3 * var[2: -1, 1:-1] -
                                     6 * var[1:-2, 1:-1] + var[0:-3, 1:-1])/6/self.dy
        else:
            par_var_x[1:-1, 1:-2] = (-1 * var[1:-1, 3:] + 6 * var[1:-1, 2:-1] -
                                3 * var[1:-1, 1:-2] -2 * var[1:-1, 0:-3])/6/self.dx
            par_var_y[1:-2, 1:-1] = (-1 * var[3:, 1:-1] + 6 * var[2: -1, 1:-1] -
                                3 * var[1:-2, 1:-1] -2 * var[0:-3, 1:-1])/6/self.dy
        return par_var_x, par_var_y
            

    # 压强2阶中心差分
    def __partial_pressure(self):
        p_press_x = np.zeros_like(self.p)
        p_press_y = np.zeros_like(self.p)
        p_press_x[1:-1, 1:-1] = (self.p[1:-1, 2:] - self.p[1:-1, 0:-2])/self.dx
        p_press_y[1:-1, 1:-1] = (self.p[2:, 1:-1] - self.p[0:-2, 1:-1])/self.dy
        return p_press_x, p_press_y
    # 扩散项压强,二阶导数
    def __partial2_pressure(self, p):
        par2_px = np.zeros_like(p)
        par2_py = np.zeros_like(p)
        par2_px[1:-1, 1:-1] = (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, 0:-2])/self.dx/self.dx
        par2_py[1:-1, 1:-1] = (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[0:-2, 1:-1])/self.dy/self.dy
        return par2_px, par2_py

    # 扩散项2阶中心差分(非压强)
    def __partial2_var(self, var):
        par2_var_x = np.zeros_like(var)
        par2_var_y = np.zeros_like(var)
        par2_var_x[1:-1, 1:-1] = (var[1:-1, 2:] - 2 * var[1:-1, 1:-1] + var[1:-1, 0:-2])/self.dx/self.dx
        par2_var_y[1:-1, 1:-1] = (var[2:, 1:-1] - 2 * var[1:-1, 1:-1] + var[0:-2, 1:-1])/self.dy/self.dy
        return par2_var_x, par2_var_y
        
    def __get_var_star(self, var):
       p_var_x_pos, p_var_y_pos = self.__partial_var(var, True) 
       p_var_x_neg, p_var_y_neg = self.__partial_var(var, False) 
       convec = 0.5 * (self.u + np.abs(self.u)) * p_var_x_pos + 0.5 * (self.u - np.abs(self.u)
                ) * p_var_x_neg + 0.5 * (self.v + np.abs(self.v)) * p_var_y_pos + 0.5 * (self.v -
                np.abs(self.v)) * p_var_y_neg
    #    convec = self.u * p_var_x_pos + self.v * p_var_y_pos
       par2_var_x, par2_var_y = self.__partial2_var(var)
       var_star = var + self.dt * ((par2_var_x + par2_var_y) * self.mu - convec)
       return var_star
    
    def __get_V_star(self):
        u_star = self.__get_var_star(self.u)
        v_star = self.__get_var_star(self.v)
        return u_star, v_star
    
    def __is_converged(self, delta):
        error = np.linalg.norm(delta, ord=np.inf)
        print(error)

        if error < self.epsion:
            return True
        return False
        
    # 求解Poisson方程,Jacobi迭代
    def __calc_p(self, u_star, v_star):
        par_u_star_x, _ = self.__partial_var(u_star, True)
        _, par_v_star_y = self.__partial_var(v_star, True)
        right_item = self.rho * (par_u_star_x + par_v_star_y) / self.dt
        
        is_converged = False
        p_ini = np.zeros_like(self.p)
        while not is_converged:
            p1 = np.zeros_like(p_ini)
            p2 = np.zeros_like(p_ini)
            p_update = np.zeros_like(p_ini)
            p1[1:-1, 1:-1] = self.dy * self.dy * (p_ini[1:-1, 2:] + p_ini[1:-1, 0:-2])
            p2[1:-1, 1:-1] = self.dx * self.dx * (p_ini[2:, 1:-1] + p_ini[0:-2, 1:-1])
            p_update[1:-1, 1:-1] = (p1[1:-1, 1:-1] + p2[1:-1, 1:-1] - right_item[1:-1, 1:-1] * self.dx * self.dx * self.dy * self.dy)/(
                2 * self.dx * self.dx + 2 * self.dy * self.dy)
            # v1, v2 = self.__partial2_pressure(p_update)
            # delta = v1 + v2 - right_item
            p_update[0, :] = 0
            p_update[-1,:] = p_update[-2, :]
            p_update[1:-1, -1] = p_update[1:-1, -2]
            p_update[1:-1, 0] = p_update[1:-1, 1]
            delta = p_update - p_ini
            is_converged = self.__is_converged(delta)
            p_ini = p_update
        self.p = p_update
        return
        
    def __step_V(self):
        u_star, v_star = self.__get_V_star()
        self.__calc_p(u_star, v_star)
        p_pressure_x, p_pressure_y = self.__partial_pressure()
        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - self.dt * p_pressure_x[1:-1, 1:-1]/self.rho
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - self.dt * p_pressure_y[1:-1, 1:-1]/self.rho
        return
        
    def get_solu(self):
        for _ in range(self.max_iter):
            old_u = self.u.copy()
            old_v = self.v.copy()
            self.__step_V()
            delta_u = old_u - self.u
            delta_v = old_v - self.v
            if self.__is_converged(delta_u) & self.__is_converged(delta_v):
                return
        print("Not converged after {} iterator!".format(self.max_iter))

        return -1

# 生成网格存储
u, v, p, dx, dy = init_con(11, 11)
u[0, :] = 2
re = 400
mu = 2 * 2 / re
prjmethod = PrjMethod(u, v, p, mu, dx, dy, dt=0.001, max_iter=1)
prjmethod.get_solu()


fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(0, 2, 11, endpoint=True)
y = np.linspace(0, 2, 11, endpoint=True)
X, Y = np.meshgrid(x, y)
ax.streamplot(X, Y, prjmethod.u, prjmethod.v)
fig.savefig('ddd.jpg', dpi=1000)