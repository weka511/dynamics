# Copyright (C) 2019 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software already_foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    return (x*y,x*x-y)

x_min = -0.25
x_max = 0.25
y_min = -0.25
y_max = 0.25
m     = 50
n     = 50
dx    = (x_max-x_min)/m
dy    = (y_max-y_min)/n

xs   = np.arange(x_min, x_max, dx)
ys   = np.arange(y_min, y_max, dy)

X, Y = np.meshgrid(xs,ys)
U, V = f(X,Y)
U    = U / np.sqrt(U**2 + V**2)
V    = V / np.sqrt(U**2 + V**2)
plt.plot(xs,              [0 for _ in xs],   color='red', label='Nullcline')
plt.plot([0 for _ in ys], ys,                color='red')
plt.plot(xs,              [x*x for x in xs], color='red')
plt.quiver(X,Y,U,V,angles='xy',units='width',   color='blue', label='Normalized derivatives', pivot='tail')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.legend()
plt.title('Exercise 6.3.10')
plt.show()