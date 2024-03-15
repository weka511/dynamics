#!/usr/bin/env python

#   Copyright (C) 2024 Simon Crase

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

'''Ergodicity of tent map'''

from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int,default=100000)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    return parser.parse_args()

def get_name_for_save(extra = None,
                      sep = '-',
                      figs = './figs'):
    '''
    Extract name for saving figure

    Parameters:
        extra    Used if we want to save more than one figure to distinguish file names
        sep      Used if we want to save more than one figure to separate extra from basic file name
        figs     Path name for saving figure

    Returns:
        A file name composed of pathname for figures, plus the base name for
        source file, with extra distinguising information if required
    '''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def f(x):
    if x<0.5: return 2*x
    if x>0.5: return 1-2*x

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)

    rng = np.random.default_rng()
    x = rng.random()
    sum = 0
    xs = []
    means = []
    for i in range(args.N):
        sum += x
        xs.append(x)
        means.append(sum/(i+1))
        x = rng.random()
    ax1.scatter(range(args.N),xs,s=1,label='$f(x)$')
    ax1.scatter(range(args.N),means,s=1,label=r'$\frac{1}{n}\sum_{i=1}^{n}f^i(x_0)$')
    ax1.set_title(__doc__)
    ax1.legend(loc='upper right',facecolor='xkcd:white')
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
