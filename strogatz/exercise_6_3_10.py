#!/usr/bin/env python

# Copyright (C) 2019-2023 Simon Crase
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

'''6.3.10 Stability of fixed point'''

from os.path import  basename,splitext
from matplotlib.pyplot import figure, show
from  phase import generate, plot_phase_portrait, plot_stability

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

def f(x,y):
    return (x*y,x*x-y)

fig = figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)

X,Y,U,V,fixed = generate(f=f,xmin=-10.0,xmax=+10.0,ymin=-10.0,ymax=+10.0,nx=256,ny=256)

plot_phase_portrait(X,Y,U,V,fixed,title=r'$\dot{x}=xy,\dot{y}=x^2-y$',ax=ax)

plot_stability(f=f,fixed=fixed,R=0.05,Limit=10.0,step=0.1,S=50,N=5000,K=100,ax=ax)
fig.suptitle('Example 6.3.10')
fig.savefig(get_name_for_save())
show()
