#!/usr/bin/env python

# Copyright (C) 2017-2023 Simon Crase

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

'''Example 6.1.7 showing Stable manifold'''

from os.path import  basename,splitext,join
from matplotlib.pyplot import figure, show
import matplotlib.colors as colors
import numpy as np
from  phase import generate, plot_phase_portrait
from rk4 import rk4, adapt

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(-x-np.exp(-y),y),nx = 256, ny  =  256,xmin = -11,xmax = 10,ymin = 0,ymax = 20)
plot_phase_portrait(X,Y,U,V,fixed,title = '$\dot{x} = -x-e^{-y},\dot{y} = y$',ax=ax)
fig.suptitle(__doc__)

eps = 1e-6
xy0 = [-1,eps]
xy = [xy0]
for j in range(10000):
    xy.append(rk4(0.01,xy[-1],adapt(f = lambda x,y:(-x-np.exp(-y),y))))

ax.plot([z[0] for z in xy],
        [z[1] for z in xy],
        c = 'r',
        label = 'Stable manifold',
        linewidth = 2)

ax.legend(loc = 'best')
fig.savefig(get_name_for_save())

show()
