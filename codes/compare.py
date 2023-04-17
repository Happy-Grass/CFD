import sod
import sod_cal_new
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 101, endpoint=True)
rho_r, u_r, p_r = sod.cal_value_all(x, 0.14)
rho_ini, u_ini, p_ini, rho, u, p = sod_cal_new.update(x, 0.14)

fig = plt.figure(figsize=(27, 9))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(x, rho_r, label='theoretical value')
ax1.plot(x, rho, label='calculated value')
ax1.text(0.1, 0.9, 't=0.14')
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\rho$')
ax1.legend()

ax2.plot(x, u_r, label='theoretical value')
ax2.plot(x, u, label='calculated value')
ax2.text(0.1, 0.9, 't=0.14')
ax2.set_xlabel('x')
ax2.set_ylabel('u')
ax2.legend()

ax3.plot(x, p_r, label='theoretical value')
ax3.plot(x, p, label='calculated value')
ax3.text(0.1, 0.9, 't=0.14')
ax3.set_xlabel('x')
ax3.set_ylabel('p')
ax3.legend()

fig.savefig('../figures/compare.png', dpi=500)
