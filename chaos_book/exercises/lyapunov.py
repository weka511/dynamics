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
from scipy.stats import linregress

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    return parser.parse_args()

def create_trajectory(N = 128,
                      delta = 0.1,
                      x0 = np.array([[1,1],[1.1,1.1]]),
                      exponents = np.array([1.0,2.0]),
                      rng = np.random.default_rng(),
                      sigma = 0.01):
    m,d = x0.shape
    assert len(exponents)==d
    ts = delta*np.array(range(N))
    trajectory = np.zeros((m,N,d))
    for i in range(m):
        for j in range(N):
            for k in range(d):
                trajectory[i,j,k] = x0[i,k] * np.exp(exponents[k]*ts[j]) * rng.normal(loc=1,scale=sigma)

    return ts,trajectory

def get_lyapunov(ts,trajectory):
    '''
    Used to calculate Lyaponov exponnent

    Parameters:
        ts          Times at which tranjctory calculated
        trajectory
    Returns:
        log_normed_diffs
        regression

    '''
    differences_from_reference = trajectory[1:,:,:] - trajectory[0,:,:]
    normed_differences = np.linalg.norm(differences_from_reference,axis=-1)
    normed_differences /= normed_differences[:,0]
    log_normed_diffs = np.log(normed_differences)
    regression = linregress(ts,log_normed_diffs)
    return log_normed_diffs,regression

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
    lyapunov,regression = get_lyapunov(ts,trajectory)

    fig = figure(figsize=(10,10))

    ax = fig.add_subplot(1,1,1)
    ax.scatter(ts,lyapunov,c='xkcd:blue',label='Lyapunov')
    ax.plot(ts,regression.intercept+regression.slope*ts,c='xkcd:red',label=f'Slope={regression.slope:.4f},r={regression.rvalue:.4f}')
    ax.legend()
    fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
