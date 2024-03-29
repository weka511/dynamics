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

'''Exercise 3.1.5 Unusual bifurcations'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def sketch_vector_field(r,f = lambda x,r: r**2 - x**2,ax=None):
    x = np.linspace(-5,5,100)
    ax.plot(x,f(x,r))
    ax.set_title(f'r={r}')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\dot{x}$')
    ax.axhline(0,c='xkcd:red',linestyle='dashed')

def sketch_vector_fields(f = lambda x,r: r**2 - x**2, rs=[0,1,2], fig=None):
    m = len(rs)
    for i in range(m):
        sketch_vector_field(r = rs[i],
                            f = f,
                            ax = fig.add_subplot(1,m,i+1))

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(10,10))
    sketch_vector_fields(fig=fig)
    fig = figure(figsize=(10,10))
    sketch_vector_fields(f = lambda x,r: r**2 + x**2,fig=fig)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
