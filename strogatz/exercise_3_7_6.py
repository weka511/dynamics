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

from os.path import  basename,splitext,join
from matplotlib.pyplot import figure, show
import numpy as np
from solver import rk4

aa = [0.5, 1,2,3,5]
bs = [0.9, 0.95, 1.0, 1.1] #[0.1,0.2,0.3]
h  = 0.1
l  = 4

colours = ['xkcd:purple',
           'xkcd:green',
           'xkcd:blue',
           'xkcd:pink',
           'xkcd:red',
           'xkcd:light blue',
           'xkcd:teal',
           'xkcd:orange'
           ]

def get_dim(N):
    m = int(np.sqrt(N))
    n = N//m
    if m*n < N:
        m += 1
    return m,n

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

us = np.arange(0,int(l),h)

fig = figure(figsize = (10,10))
for j,b in enumerate(bs):
    ax = fig.add_subplot(*get_dim(len(bs)),j+1)
    ax.plot(us,np.exp(-us),
         c = 'xkcd:black',
         label = r'$e^{-u}$')
    for i,a in enumerate(aa):
        ax.plot(us, a - b *us,
             c = colours[i],
             label = f'a={a},b={b}')
    ax.legend()
    ax.set_xlabel('u')
    ax.grid(True)

fig.suptitle(__doc__)
fig.savefig(get_name_for_save(extra=1))

def f(y,k=0.0001,l=0.001):
    return np.array([
        -k * y[0] * y[1],
        k * y[0] * y[1] - l* y[1],
        l* y[1]
    ])

m = 100000
y = np.zeros((m+1,3))
y[0,0] = 1000
y[0,1] = 1
h = 0.01
for i in range(1,m+1):
    y[i,:] = rk4(h, y[i-1,:], f)

fig = figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(y[:,0],label='x')
ax.plot(y[:,1],label='y')
ax.plot(y[:,2],label='z')
ax.legend()
ax.set_title(__doc__)
fig.savefig(get_name_for_save(extra=2))

show()
