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

# Stability of fixed point

import sys
sys.path.append('../')
import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase,numpy as np,rk4,utilities

def f(x,y):
    return (x*y,x*x-y)

cs            = ['r','b','g','m','c','y']
linestyles    = ['-', '--', '-.', ':']

X,Y,U,V,fixed = phase.generate(f=f,xmin=-10.0,xmax=+10.0,ymin=-10.0,ymax=+10.0)
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=xy,\dot{y}=x^2-y$',suptitle='Example 6.3.10') 

starts        = [ utilities.direct_sphere(d=2,R=1) for i in range(len(cs)*len(linestyles))]
for xy0,i in zip(starts,range(len(starts))):
    xy=[xy0]
    for j in range(1000):
        xy.append(rk4.rk4(0.1,xy[-1],phase.adapt(f=f)))
    plt.plot([z[0] for z in xy],
             [z[1] for z in xy],
             c         = cs[i%len(cs)],
             linestyle = linestyles[i//len(cs)],
             label     = '({0:.3f},{1:.3f})'.format(xy0[0],xy0[1]),linewidth=3)

plt.show()