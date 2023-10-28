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

'''
    Exercise 6.3 from Strogatz
    Plot phase portraits for a number of ODEs
'''

from  matplotlib.pyplot import figure, show
import matplotlib.colors as colors
import numpy as np
from phase import generate,plot_phase_portrait,adapt
from rk4 import rk4
import utilities

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed=generate(f=lambda x,y:(x-y,x*x-4))
plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=x-y,\dot{y}=x^2-4$',ax=ax)
fig.suptitle('Example 6.3.1')

fig = figure()
ax = fig.add_subplot(1,1,1)

def f(x,y,direction=1):
    return (direction*(y*y*y-4*x),direction*(y*y*y-y-3*x))

X,Y,U,V,fixed=generate(f=f,nx=256, ny = 256,xmin=-100,xmax=100,ymin=-100,ymax=100)

plot_phase_portrait(X,Y,U,V,fixed,title='$\dot{x}=y^3-4x,\dot{y}=y^3-y-3x$',ax=ax)

cs = ['r','b','g','m','c','y']
starts=[ (0,25*i) for i in range(-5,6)]
for xy0,i in zip(starts,range(len(starts))):
    xy=[xy0]
    for j in range(100000):
        xy.append(rk4(0.0001,xy[-1],adapt(f=lambda x,y:f(x,y,-1))))
    ax.plot([z[0] for z in xy],
             [z[1] for z in xy],
             c=cs[i%len(cs)],
             label='({0:.3f},{1:.3f})'.format(xy0[0],xy0[1]),linewidth=3)

ax.legend(loc='best')
fig.suptitle('Example 6.3.9')

show()
