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

'''Exercose 10.2 The Burke-Shaw System'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from scipy.integrate import solve_ivp

s = 10.5
v = 4.272

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--eps',  default = 0.00001,type=float)
    parser.add_argument('--fp', default=0, type=int)
    return parser.parse_args()

def Velocity(t,ssp):
    x = ssp[0]
    y = ssp[1]
    z = ssp[2]
    return np.array([-s*(x + y),
                     -y - s*x*z,
                     s*x*y + v])

def StabilityMatrix(ssp):
    x = ssp[0]
    y = ssp[1]
    z = ssp[2]
    return np.array([[-s, -s, 0],
                     [-s*z, -1, -s*x],
                     [s*y, s*x, 0]])

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

def get_fp():
    x = np.sqrt(v/s)
    y = -x
    z = 1/s
    return np.array([[x,y,z],[-x,-y,z]])

if __name__=='__main__':
    start  = time()
    args = parse_args()

    eq0 = get_fp()[args.fp]
    Aeq0 = StabilityMatrix(eq0)
    _, eigenVectors = np.linalg.eig(Aeq0)
    v1 = np.real(eigenVectors[:, 0])
    v1 = v1 / np.linalg.norm(v1)
    solution = solve_ivp(Velocity,(0,1000),eq0 + args.eps * v1)
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.scatter(solution.y[0,:], solution.y[1,:], solution.y[2,:],s=1)
    ax1.set_title(f'Burke-Shaw s={s}, v={v}')
    fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
