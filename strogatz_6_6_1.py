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

import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase

        
def f(x,y):
    return y*(1-y*y),-x-y*y

X,Y,U,V=phase.generate(f=f)
phase.plot_phase_portrait(X,Y,U,V,title=r'$\dot{x}=y-y^3,\dot{y}=-x-y^2$',suptitle='Example 6.6.1')  
plt.show()