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

# Exercise 6.1 from Strogatz
# Plot phase porttraits for a number of ODEs

import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase,numpy as np,rk4



def f(x,y):
    return y,-x+(1-x*x-y*y)*y

X,Y,U,V,fixed=phase.generate(f=f,nx=256, ny = 256,xmin=-1,xmax=1,ymin=-1,ymax=1)

phase.plot_phase_portrait(X,Y,U,V,fixed,title='$\dot{x}=y,\dot{y}=x+(1-x^2-y^2)y$',suptitle='Exercise 6.2.1')

cs = ['r','b','g','m','c','y']
starts=[(0.5,0),(0.6,0),(0.4,0)]
for xy0,i in zip(starts,range(len(starts))):
    xy=[xy0]
    for j in range(100000):
        xy.append(rk4.rk4(0.01,xy[-1],phase.adapt(f=f)))
    plt.plot([z[0] for z in xy],
                    [z[1] for z in xy],
                    c=cs[i%len(cs)],
                    linewidth=1)  

leg=plt.legend(loc='best')
if leg:
    leg.draggable()
    

    
plt.show()