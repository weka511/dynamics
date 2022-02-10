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

# Exercise 6.4 from Strogatz
# Plot phase porttraits for a number of ODEs

import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase,numpy as np,rk4,utilities

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(3-x-y),y*(2-x-y)),nx=256,ny=256,xmin=-0.5,xmax=3.5,ymin=-0.5,ymax=3.5)
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(3-x-y),\dot{y}=y(2-x-y)$',suptitle='Example 6.4.1') 
plt.figure()

f=lambda x,y:(x*(3-2*x-y),y*(2-x-y))

X,Y,U,V,fixed=phase.generate(f=f,nx=256,ny=256,xmin=-0.5,xmax=3.5,ymin=-0.5,ymax=3.5)
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(3-2x-y),\dot{y}=y(2-x-y)$',suptitle='Example 6.4.2') 

cs = ['r','b','g','m','c','y']    
starts=[ (0.01,0.1),(1.05,0.1),(0.1,2.1),(1.1,0.1),(10,10),(0.1,1.9)]
for xy0,i in zip(starts,range(len(starts))):
    xy=[xy0]
    for j in range(1000000):
        xy.append(rk4.rk4(0.0001,xy[-1],phase.adapt(f=f)))
    plt.plot([z[0] for z in xy],
             [z[1] for z in xy],
             c=cs[i%len(cs)],
             label='({0:.3f},{1:.3f})'.format(xy0[0],xy0[1]),linewidth=3)

leg=plt.legend(loc='best')
if leg:
    leg.draggable()
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(3-2*x-2*y),y*(2-x-y)),nx=256,ny=256,xmin=-0.5,xmax=3.5,ymin=-0.5,ymax=3.5)
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(3-2x-2y),\dot{y}=y(2-x-y)$',suptitle='Example 6.4.3') 

     
plt.show()