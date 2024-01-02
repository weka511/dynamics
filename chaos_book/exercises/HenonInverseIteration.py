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

'''Chaosbook Exercise 7.2: Inverse Iteration Method for the Hénon repeller'''


from argparse import ArgumentParser, ArgumentTypeError
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

a = 6

def sign(s):
    converted = int(s)
    if converted==0: return -1
    elif converted==1: return 1
    else: raise ArgumentTypeError(f'{s} should be 0 or 1')

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--N', type = int, default = 1000)
    parser.add_argument('--M', type = int, default = 100)
    parser.add_argument('--S', type = sign, nargs = '+')
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

def get_sign(S,i):
    return S[i%len(S)]

if __name__=='__main__':
    start  = time()
    args = parse_args()
    rng = np.random.default_rng()
    X = rng.uniform(0,0.5,2*args.M + args.N)
    for i in range(args.M):
        for j in range(1,X.size-1):
            X[j] =   get_sign(args.S,j) * np.sqrt((1-X[j-1]-X[j+1])/a)

    Cycle = X[args.M:args.M+len(args.S)]
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(range(args.N),X[args.M:-args.M],s=1)
    ax1.set_title(f'{args.S} {Cycle.sum()}')
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
