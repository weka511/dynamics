#!/usr/bin/env python

# Copyright (C) 2019 Greenweaves Software Limited

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
# Plot phase portraits for a number of ODEs

import sys
sys.path.append('../')

import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase,numpy as np,rk4,utilities
for mu in [1,2,3]:
    plt.figure()
    X,Y,U,V,fixed=phase.generate(f=lambda x,y:(x*(1-y),mu*y*(x-1)),nx=64,ny=64,xmin=-0.05,xmax=3.0,ymin=-0.05,ymax=3.0)
    phase.plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x(1-y),\dot{y}=\mu y(y-1)$',suptitle='Example 6.5.19 - Lotka-Volterra')




plt.show()
