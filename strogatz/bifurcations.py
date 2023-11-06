#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Model to display fixed points'''


from argparse import ArgumentParser
from os.path import  basename,splitext
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('r', default=[1,2,3], type=float,nargs='+')
    parser.add_argument('--show', default = False, action='store_true')
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

def sketch_vector_field(r,
                        f = lambda x,r: x*(r-np.exp(x)),
                        df = lambda x,r: r-np.exp(x)-x*np.exp(x),
                        fixed_points = [],
                        x = np.linspace(-1,1,100),
                        ax = None):
    def mark_fixed_point(x):
        if df(x,r) > 0:
            facecolors= 'none'
            label = 'unstable'
        else:
            facecolors= 'xkcd:black'
            label = 'stable'
        ax.scatter(x,0,s=80, facecolors=facecolors, edgecolors='xkcd:black',label=label)

    def legend_without_duplicate_labels(ax):
        '''
        Suppress duplicate items in legend. Snarfed from
        https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
        '''
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    ax.plot(x,f(x,r))
    for x0 in fixed_points:
        mark_fixed_point(x0)
        ax.axvline(x0,c='xkcd:magenta',linestyle='dotted')
    ax.set_title(f'r={r}')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\dot{x}$')
    ax.axhline(0,c='xkcd:red',linestyle='dashed')
    legend_without_duplicate_labels(ax)

