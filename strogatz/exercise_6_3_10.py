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
import  matplotlib.pyplot as plt,phase

def f(x,y):
    return (x*y,x*x-y)

X,Y,U,V,fixed_points = phase.generate(f=f,xmin=-10.0,xmax=+10.0,ymin=-10.0,ymax=+10.0)
phase.plot_phase_portrait(X,Y,U,V,fixed_points,title=r'$\dot{x}=xy,\dot{y}=x^2-y$',suptitle='Example 6.3.10') 

phase.plot_stability(f=f,fixed_points=fixed_points)
plt.legend()
plt.show()