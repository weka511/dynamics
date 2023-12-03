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

'''Calculate Lyapunov expenent'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    return parser.parse_args()

def create_trajectory(m = 2,
                      N = 16,
                      delta = 0.1):
    ts = delta*np.array(range(N))
    trajectory = np.zeros((m,N,1))
    trajectory[0,0,0] = 1
    trajectory[1,0,0] = 1.1
    for i in range(m):
        for j in range(1,N):
            trajectory[i,j,0] = trajectory[i,0,0] * np.exp(delta * j)
    return ts,trajectory

def get_lyapunov(trajectory):
    '''Used to calculate Lyaponov exponnent'''
    differences_from_reference = trajectory[1:,:,:] - trajectory[0,:,:]
    normed_differences = np.linalg.norm(differences_from_reference,axis=-1)
    normed_differences /= normed_differences[:,0]
    log_normed_diffs = np.log(normed_differences)
    return log_normed_diffs

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

if __name__=='__main__':
    start  = time()
    args = parse_args()

    ts,trajectory = create_trajectory()
    lyapunov = get_lyapunov(trajectory)

    fig = figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1)
    ax1.scatter(ts,trajectory[0,:,0])
    ax1.scatter(ts,trajectory[1,:,0])
    ax2 = fig.add_subplot(2,1,2)
    ax2.scatter(ts,lyapunov)
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
