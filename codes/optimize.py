import numpy as np
from matplotlib import pyplot as plt

a = np.arange(0, np.pi + 0.01, 0.01)
print(a)
t = np.cos(a)
f_t = 4 * t**3 - 12 * t**2 + 12 * t + 7
k_r = -1 / 540 * f_t
k_i = 1 / 540 * np.sin(3 * a) - 94 / 540 * np.sin(
    2 * a) + 725 / 540 * np.sin(a)

fig = plt.figure(dpi=80)
ax = fig.add_subplot(111)
# ax.plot(a, k_r)
ax.plot(a, k_i)
ax.plot(a, a)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("$k_i$")
# ax.set_ylabel('$k_r$')
# plt.savefig('k_i.png', dpi=500)
plt.show()
