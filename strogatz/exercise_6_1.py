#!/usr/bin/env python

# Copyright (C) 2017-2023 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

'''
    Exercise 6.1 from Strogatz
    Plot phase portraits for a number of ODEs
'''

from matplotlib.pyplot import figure, show
import matplotlib.colors as colors
import numpy as np
from  phase import generate, plot_phase_portrait, adapt
from rk4 import rk4

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(x-y,1-np.exp(x)))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = x-y,\dot{y} = 1-e^x$',ax=ax)
fig.suptitle('Example 6.1.1')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(x-x**3,-y))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = x-x^3,\dot{y} = -y$',ax=ax)
fig.suptitle('Example 6.1.2')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,_ = generate(f = lambda x,y:(x*(x-y),y*(2*x-y)))
plot_phase_portrait(X,Y,U,V,[(0,0)],title = r'$\dot{x} = x(x-y),\dot{y} = y*(2x-y)$',ax=ax)
fig.suptitle('Example 6.1.3')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(y,x*(1+y)-1))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = y,\dot{y} = x(1+y)-1$',ax=ax)
fig.suptitle('Example 6.1.4')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(x*(2-x-y),x-y))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = x(2-x-y),\dot{y} = x-y$',ax=ax)
fig.suptitle('Example 6.1.5')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(x*x-y,x-y))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = x^2-y,\dot{y} = x-y$',ax=ax)
fig.suptitle('Example 6.1.6')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(-x-np.exp(-y),y),nx = 256, ny  =  256,xmin = -11,xmax = 10,ymin = 0,ymax = 20)
plot_phase_portrait(X,Y,U,V,fixed,title = '$\dot{x} = -x-e^{-y},\dot{y} = y$',ax=ax)
fig.suptitle('Example 6.1.7 showing Stable manifold')

eps = 1e-6
xy0 = [-1,eps]
xy = [xy0]
for j in range(10000):
    xy.append(rk4(0.01,xy[-1],adapt(f = lambda x,y:(-x-np.exp(-y),y))))

ax.plot([z[0] for z in xy],
        [z[1] for z in xy],
        c = 'r',
        label = 'Stable manifold',
        linewidth = 2)

ax.legend(loc = 'best')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(x,-x+y*(1-x*x)))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = x,\dot{y} = -x+y(1-x^2)$',ax=ax)
fig.suptitle('Example 6.1.8 - van de Pol')

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(y+y*y,y*y-x*x))
plot_phase_portrait(X,Y,U,V,[(0,0)],title = r'$\dot{x} = 2xy,\dot{y} = y^2-x^2$',ax=ax)
fig.suptitle('Example 6.1.9 - Dipole fixed point')


fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(y+y*y,-x/2+y/5-x*y+6*y*y/5))
plot_phase_portrait(X,Y,U,V,fixed,
                          title = r'$\dot{x} = y+y^2,\dot{y} = -\frac{x}{2}+\frac{y}{5}-xy+\frac{6}{5}y^2$',
                          ax = ax)
fig.suptitle ('Example 6.1.10 - Two eyed monster')


fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(y+y*y,-x+y/5-x*y+6*y*y/5))
plot_phase_portrait(X,Y,U,V,fixed,
                          title = r'$\dot{x} = y+y^2,\dot{y} = -x+\frac{y}{5}-xy+\frac{6}{5}y^2$',
                          ax=ax)
fig.suptitle('Example 6.1.11 - Parrot')

show()
