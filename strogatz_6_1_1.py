# Copyright (C) 2017 Greenweaves Software Pty Ltd

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

import numpy as np, matplotlib.pyplot as plt,matplotlib.colors as colors

from scipy.integrate import odeint
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def ex(x,y):
    return x+np.exp(-y),-y

def ff(x,t,f=ex):
    u,v=f(x[0],x[1])
    return [u]+[v]

nx, ny = 256, 256
x = np.linspace(-5, 5, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y)
U,V=ex(X,Y)

@np.vectorize
def nullclines(u,v):
    if u<0:
        if v<0:
            return 4
        else:
            return 1
    else:
        if v<0:
            return 2
        else:
            return 3
        
plt.pcolor(X,Y,nullclines(U,V),cmap=plt.cm.inferno)
plt.streamplot(X, Y, U, V, color=U, linewidth=1, cmap=plt.cm.inferno)
plt.colorbar()
t = np.linspace(0, 10, 101)
cs = ['r','b','g','m','c','y']
i=0

for xy0 in [[-2,3],[-0.5,3],[-1.1,3],[-2,-3],[-2,-3],[-2,-3]]:
    xy = odeint(ff, xy0, t)
    plt.plot(xy[:,0],xy[:,1],c=cs[i%len(cs)],label='({0},{1})'.format(xy0[0],xy0[1]),linewidth=3)
    plt.xlim(-5,5)
    plt.ylim(-3,3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Example 6.1.1')
    i+=1
plt.legend(loc='best')    
plt.show()