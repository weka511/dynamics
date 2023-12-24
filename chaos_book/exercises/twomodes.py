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

'''Chaosbook Example 12.8: Visualize two-modes flow'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp

mu1 = -2.8
a2 = -2.66
c1 = -7.75

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--T', type=float, default=10000)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    return parser.parse_args()

ix1 = 0
iy1 = 1
ix2 = 2
iy2 = 3

def Velocity(t,ssp):
    x1 = ssp[ix1]
    y1 = ssp[iy1]
    x2 = ssp[ix2]
    y2 = ssp[iy2]
    r2 = x1**2 + y1**2
    return np.array([
        (mu1-r2)*x1 + c1*(x1*x2 + y1*y2),
        (mu1-r2)*y1 + c1*(x1*y2 - x2*y1),
        x2 + y2 + x1**2 - y1**2 +a2*x2*r2 ,
        -x2 + y2+ 2*x1*y1 + a2*y2*r2
    ])

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

if __name__=='__main__':
    start  = time()
    args = parse_args()

    rng = np.random.default_rng()
    ssp = rng.uniform(-1,1,4)
    solution = solve_ivp(Velocity, (0,args.T), ssp)

    fig = figure(figsize=(12,12))

    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(solution.y[ix2,:],solution.y[ix1,:],solution.y[iy2,:],
                c = solution.y[iy1,:],
                s = 1,
                cmap = 'viridis',
                label = 'Trajectory')
    ax1.scatter(solution.y[ix2,0],solution.y[ix1,0],solution.y[iy2,0],
                marker = 'X',
                label = 'Start',
                c = 'xkcd:red')
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1),
                                cmap = 'viridis'),
                 ax = ax1,
                 label = 'y1')
    ax1.set_xlabel('x2')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('y2')
    ax1.legend()
    ax1.set_title(f'Figure 12.1 T={args.T}')

    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.scatter(solution.y[ix1,:],solution.y[iy1,:],solution.y[iy2,:],
                c = solution.y[ix2,:],
                s = 1,
                cmap = 'viridis',
                label = 'Trajectory')
    ax2.scatter(solution.y[ix1,0],solution.y[iy1,0],solution.y[iy2,0],
                marker = 'X',
                label = 'Start',
                c = 'xkcd:red')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('y1')
    ax2.set_zlabel('y2')
    ax2.legend()
    ax2.set_title(f'Exercise 12.7 T={args.T}')

    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
