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
    parser.add_argument('--a', default= 1.4, type=float)
    parser.add_argument('--b', default= 0.3, type=float)
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

def get_trajectory(a = 1.4,
                   b = 0.3,
                   N = 100000,
                   N0 = 0):
    trajectory = np.zeros((args.m,N,2))
    for i in range(args.m):
        trajectory[i,0,:] = args.epsilon * rng.random(2)
        k = 0
        for j in range(1,N0):
            trajectory[i,1,:] = henon( trajectory[i,k,:],a=a,b=b)
            k = 1
        for j in range(k+1,N):
            trajectory[i,j,:] = henon( trajectory[i,j-1,:],a=a,b=b)

    return trajectory

def get_lyapunov(trajectory):
    diffs = trajectory[1:,:,:] - trajectory[0,:,:]
    normed_diffs = np.linalg.norm(diffs,axis=-1)
    normed_diffs /= normed_diffs[0]
    log_normed_diffs = np.log(normed_diffs)
    return log_normed_diffs

if __name__=='__main__':
    start  = time()
    rng = np.random.default_rng()
    args = parse_args()

    trajectory = get_trajectory(a=args.a,b=args.b,N=args.N,N0=args.N0)
    log_normed_diffs = get_lyapunov(trajectory)

    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,2,1)
    for i in range(args.m):
        ax1.scatter(trajectory[i,:,0],trajectory[i,:,1],s=1)
        ax1.scatter(trajectory[i,-1,0],trajectory[i,-1,1],marker='+',s=100)

    ax2 = fig.add_subplot(1,2,2)
    m,n = log_normed_diffs.shape
    for i in range(m):
        ax2.scatter(list(range(n)),log_normed_diffs[i,:],s=1)

    fig.suptitle(f'a={args.a}, b=args.b, Burn in = {args.N0:,}')
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
