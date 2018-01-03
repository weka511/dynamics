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

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x-y,1-np.exp(x)))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x-y,\dot{y}=1-e^x$',suptitle='Example 6.1.1') 
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x-x**3,-y))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x-x^3,\dot{y}=-y$',suptitle='Example 6.1.2') 
plt.figure()

X,Y,U,V,_=phase.generate(f=lambda x,y:(x*(x-y),y*(2*x-y)))
phase.plot_phase_portrait(X,Y,U,V,[(0,0)],title=r'$\dot{x}=x(x-y),\dot{y}=y*(2x-y)$',suptitle='Example 6.1.3') 
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(y,x*(1+y)-1))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=y,\dot{y}=x(1+y)-1$',suptitle='Example 6.1.4') 
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(2-x-y),x-y))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(2-x-y),\dot{y}=x-y$',suptitle='Example 6.1.5') 
plt.figure() 

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*x-y,x-y))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x^2-y,\dot{y}=x-y$',suptitle='Example 6.1.6') 
plt.figure()

def f(x,y):
    return -x-np.exp(-y),y

X,Y,U,V,fixed=phase.generate(f=f,nx=256, ny = 256,xmin=-11,xmax=10,ymin=0,ymax=20)

phase.plot_phase_portrait(X,Y,U,V,fixed,title='$\dot{x}=-x-e^{-y},\dot{y}=y$',suptitle='Example 6.1.7')

eps=1e-6
xy0=[-1,eps]
xy=[xy0]
for j in range(10000):
    xy.append(rk4.rk4(0.01,xy[-1],phase.adapt(f=f)))
plt.plot([z[0] for z in xy],
                [z[1] for z in xy],
                c='r',
                label='Stable manifold',
                linewidth=2)  

leg=plt.legend(loc='best')
if leg:
    leg.draggable()
    
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x,-x+y*(1-x*x)))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x,\dot{y}=-x+y(1-x^2)$',suptitle='Example 6.1.8 - van de Pol')
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(y+y*y,y*y-x*x))
phase.plot_phase_portrait(X,Y,U,V,[(0,0)],title=r'$\dot{x}=2xy,\dot{y}=y^2-x^2$',suptitle='Example 6.1.9 - Dipole fixed point')
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(y+y*y,-x/2+y/5-x*y+6*y*y/5))
phase.plot_phase_portrait(X,Y,U,V,fixed,
                          title=r'$\dot{x}=y+y^2,\dot{y}=-\frac{x}{2}+\frac{y}{5}-xy+\frac{6}{5}y^2$',
                          suptitle='Example 6.1.10 - Two eyed monster')
plt.figure()
X,Y,U,V,fixed=phase.generate(f=lambda x,y:(y+y*y,-x+y/5-x*y+6*y*y/5))
phase.plot_phase_portrait(X,Y,U,V,fixed,
                          title=r'$\dot{x}=y+y^2,\dot{y}=-x+\frac{y}{5}-xy+\frac{6}{5}y^2$',
                          suptitle='Example 6.1.11 - Parrot')


    
plt.show()