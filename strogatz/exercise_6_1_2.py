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
    Exercise 6.1 from Strogatz
    Plot phase portraits for a number of ODEs
'''

from os.path import  basename,splitext
from matplotlib.pyplot import figure, show
import matplotlib.colors as colors
import numpy as np
from  phase import generate, plot_phase_portrait
from rk4 import rk4, adapt

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

fig = figure()
ax = fig.add_subplot(1,1,1)
X,Y,U,V,fixed = generate(f = lambda x,y:(x-x**3,-y))
plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = x-x^3,\dot{y} = -y$',ax=ax)
fig.suptitle('Example 6.1.2')
fig.savefig(get_name_for_save())
show()
