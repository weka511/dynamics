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

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x-y,x*x-4))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x-y,\dot{y}=x^2-4$',suptitle='Example 6.3.1') 
plt.figure()

import utilities,rk4

def f(x,y):
    return (y*y*y-4*x,y*y*y-y-3*x)

X,Y,U,V,fixed=phase.generate(f=f,nx=256, ny = 256,xmin=-20,xmax=20,ymin=-20,ymax=20)

phase.plot_phase_portrait(X,Y,U,V,fixed,title='$\dot{x}=y^3-4x,\dot{y}=y^3-y-3x$',suptitle='Example 6.3.9')

cs = ['r','b','g','m','c','y']    
starts=[ utilities.direct_sphere(d=2,R=10) for i in range(6)]
for xy0,i in zip(starts,range(len(starts))):
    xy=[xy0]
    for j in range(100000):
        xy.append(rk4.rk4(0.0001,xy[-1],phase.adapt(f=f)))
    plt.plot([z[0] for z in xy],
             [z[1] for z in xy],
             c=cs[i%len(cs)],
             label='({0:.3f},{1:.3f})'.format(xy0[0],xy0[1]),linewidth=3)

leg=plt.legend(loc='best')
if leg:
    leg.draggable()
     
plt.show()