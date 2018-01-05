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

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(3-x-y),y*(2-x-y)))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(3-x-y),\dot{y}=y(2-x,y)$',suptitle='Example 6.4.1') 
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(3-2*x-y),y*(2-x-y)))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(3-2x-y),\dot{y}=y(2-x,y)$',suptitle='Example 6.4.2') 
plt.figure()

X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(3-2*x-2*y),y*(2-x-y)))
phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(3-2x-2y),\dot{y}=y(2-x,y)$',suptitle='Example 6.4.3') 

     
plt.show()