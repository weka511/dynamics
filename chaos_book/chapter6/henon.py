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

'''Template for python script for dynamics'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--N',default = 100000, type=int)
    parser.add_argument('--N0',default = 1000, type=int)
    parser.add_argument('--m',default = 11, type=int)
    parser.add_argument('--epsilon', default= 0.0001, type=float)
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
        source file, with extra ditinguising information if required
    '''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def henon(x=0,a=1.4,b=0.3):
    return np.array([1 - a*x[0]**2 + x[1], b*x[0]])

def plot_henon(a=1.4,b=0.3,N=100000,N0=0,ax=None):
    for i in range(args.m):
        trajectory=np.zeros((N,2))
        trajectory[0,:] = args.epsilon * rng.random(2)
        for j in range(1,N0):
            trajectory[0,:] = henon( trajectory[0,:],a=a,b=b)
        for j in range(1,N):
            trajectory[j,:] = henon( trajectory[j-1,:],a=a,b=b)
        ax.scatter(trajectory[:,0],trajectory[:,1],s=1)
        ax.scatter(trajectory[-1,0],trajectory[-1,1],marker='+',s=100)
    ax.set_title(f'a={a},Burn in = {N0:,}')

if __name__=='__main__':
    start  = time()
    rng = np.random.default_rng()
    args = parse_args()
    fig = figure(figsize=(12,12))
    plot_henon(a=1.4,b=0.3,N=args.N,ax=fig.add_subplot(2,2,1))
    plot_henon(a=1.4,b=0.3,N=args.N,N0=args.N0,ax=fig.add_subplot(2,2,3))
    plot_henon(a=1.39945219,b=0.3,N=args.N,ax=fig.add_subplot(2,2,2))
    plot_henon(a=1.39945219,b=0.3,N=args.N,N0=args.N0,ax=fig.add_subplot(2,2,4))
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
