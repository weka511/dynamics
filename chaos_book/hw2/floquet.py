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
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/

# Q1.2 A limit cycle with analytic Floquet exponent

import sys
sys.path.append('../../')
import  matplotlib.pyplot as plt,matplotlib.colors as colors,phase,numpy as np,rk4

def floquet(q,p):
    return (p+q*(1-q*q-p*p),-q+p*(1-q*q-p*p))

plt.figure(figsize=(20,20))
X,Y,U,V,fixed_points=phase.generate(f=floquet,
                             nx=256, ny = 256,
                             xmin=-1.2,xmax=1.2,ymin=-1.2,ymax=1.2)

phase.plot_phase_portrait(X,Y,U,V,fixed_points,title=r'$\dot{p}=p+q(1-q^2-p^2),\dot{q}=-q+p(1-q^2-p^2)$',
                          suptitle='Q1.2 A limit cycle with analytic Floquet exponent.'
                          ' (ChaosBook.org version 14.5.7, exercise 5.1)',
                          xlabel='$p$',
                          ylabel='$q$')

phase.plot_stability(f=floquet,fixed_points=fixed_points,Limit=0.99,step=0.1,N=50000,R=0.01)

plt.savefig('floquet.png')

plt.show()
