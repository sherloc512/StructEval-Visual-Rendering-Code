from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)

# put 0s on the y-axis, and put the y axis on the z-axis
ax.plot(xs=self.grid_s, ys=self.grid_vu, zs=self.p[2][eye(self.grid_s.shape[0], dtype=bool)], zdir='z', label='ys=0, zdir=z')
plt.show()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure(figsize=(14,8))
ax = fig.gca(projection='3d')

z = self.p[2][eye(self.grid_s.shape[0], dtype=bool)]
x = self.grid_s 
y = self.grid_vu

ax.plot(x, y, z, label='new copy price')
ax.legend()
ax.set_zlabel('new copy price')
ax.set_xlabel('s (cumulative sales)')
ax.set_ylabel('vu (remaining demand)')

plt.show()

# surface plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

self = multiperiod
fig = plt.figure(figsize=(14,8))
ax = fig.gca(projection='3d')
X = self.grid_s
Y = self.grid_vu
X, Y = np.meshgrid(X, Y)
Z = self.p[5].T
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_zlabel('new copy price')
ax.set_xlabel('s (cumulative sales)')
ax.set_ylabel('vu (remaining demand)')

plt.show()

