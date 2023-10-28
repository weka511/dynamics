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

'''3.7.6 Kermack & McKendrick model of an epidemic'''

from os.path import  basename,splitext
from matplotlib.pyplot import figure, show
import numpy as np

aa = [1,2,3]
bs = [0.1,0.2,0.3]
h  = 0.1
l  = 10.0
colours = ['xkcd:purple',
           'xkcd:green',
           'xkcd:blue',
           'xkcd:pink',
           'xkcd:brown',
           'xkcd:red',
           'xkcd:light blue',
           'xkcd:teal',
           'xkcd:orange'
           ]

def get_name_for_save():
    '''Extract name for saving figure'''
    return splitext(basename(__file__))[0]

us = np.array([h*u for u in range(0,int(l/h))])

fig = figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(us,np.exp(-us),
     c = 'xkcd:black',
     label = r'$e^{-u}$')

for i,a in enumerate(aa):
    for j,b in enumerate(bs):
        u1s = a - b *us
        ax.plot(us,u1s,
             c = colours[len(bs)*i+j],
             label = f'a={a},b={b}')

ax.set_title(__doc__)
ax.grid(True)
ax.legend()
fig.savefig(get_name_for_save())
show()
