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

import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase,numpy as np


def f1(x,y):
    return x-y,1-np.exp(x)

X,Y,U,V=phase.generate(f=f1)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=x-y,\dot{y}=1-e^x$',suptitle='Example 6.1.1') 
plt.figure()

def f2(x,y):
    return x-x**3,-y

X,Y,U,V=phase.generate(f=f2)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=x-x^3,\dot{y}=-y$',suptitle='Example 6.1.2') 
plt.figure()

def f3(x,y):
    return x*(x-y),y*(2*x-y)

X,Y,U,V=phase.generate(f=f3)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=x(x-y),\dot{y}=y*(2x-y)$',suptitle='Example 6.1.3') 
plt.figure()

def f4(x,y):
    return y,x*(1+y)-1

X,Y,U,V=phase.generate(f=f4)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=y,\dot{y}=x(1+y)-1$',suptitle='Example 6.1.4') 
plt.figure()

def f5(x,y):
    return x*(2-x-y),x-y

X,Y,U,V=phase.generate(f=f5)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=x(2-x-y),\dot{y}=x-y$',suptitle='Example 6.1.5') 
plt.figure()

def f6(x,y):
    return x*x-y,x-y

X,Y,U,V=phase.generate(f=f6)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=x^2-y,\dot{y}=x-y$',suptitle='Example 6.1.6') 


plt.show()