#!/usr/bin/env python

# Copyright (C) 2017-2019 Greenweaves Software Limited

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

'''
Exercise 6.2 from Strogatz
Plot phase portraits for a number of ODEs
'''

from os.path import  basename,splitext,join
from matplotlib.pyplot import figure, show
import matplotlib.colors as colors
import numpy as np
from phase import generate,plot_phase_portrait
from rk4 import rk4, adapt

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def f(x,y):
    return y,-x + (1 - x**2 -y**2)*y

X,Y,U,V,fixed=generate(f=f,nx=256, ny = 256,xmin=-1,xmax=1,ymin=-1,ymax=1)
fig = figure()
ax = fig.add_subplot(1,1,1)
plot_phase_portrait(X,Y,U,V,fixed,title='$\dot{x}=y,\dot{y}=x+(1-x^2-y^2)y$',ax=ax)
fig.suptitle('Exercise 6.2.1')
cs = ['r','b','g','m','c','y']
starts=[(0.5,0),(0.6,0),(0.4,0)]

for xy0,i in zip(starts,range(len(starts))):
    xy=[xy0]
    for j in range(100000):
        xy.append(rk4(0.01,xy[-1],adapt(f=f)))
    ax.plot([z[0] for z in xy],
            [z[1] for z in xy],
            c = cs[i%len(cs)],
            linewidth = 1)

fig.savefig(get_name_for_save())
show()
