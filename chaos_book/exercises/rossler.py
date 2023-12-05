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
from henon import evolve
from solver import rk4



def Velocity(ssp,
             a = 0.2,
             b = 0.2,
             c = 5.7):
    '''
    Velocity function for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]

    Outputs:
        vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
    '''

    x, y, z = ssp

    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)

    return np.array([dxdt, dydt, dzdt], float)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--N', default = 100000, type=int)
    parser.add_argument('--N0', default = None, type=int)
    parser.add_argument('--epsilon', default = 0.000001, type=float)
    parser.add_argument('--a', default= 1.4, type=float)
    parser.add_argument('--b', default= 0.3, type=float)
    parser.add_argument('--xtol', default= 1.0, type=float)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--seed', default = None, type=int)
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

if __name__=='__main__':
    start  = time()
    args = parse_args()
    delta_t = 0.1
    rng = np.random.default_rng(args.seed)
    x0 = np.array([1,1,1])
    x1 = x0 + rng.uniform(0.0,args.epsilon,size=3)
    trajectory1,trajectory2,lyapunov_lambda,lyapunov = evolve(x0,x1,
                                                              mapping = lambda x:rk4(delta_t,x,Velocity),
                                                              N = 10000,
                                                              xtol = 1.0,
                                                              delta_t = delta_t)
    lambdas = list(zip(*lyapunov))
    fig = figure()
    ax1  = fig.add_subplot(2,1,1,projection='3d')
    ax1.plot(trajectory1[:,0],trajectory1[:,1],trajectory1[:,2])
    ax1.plot(trajectory2[:,0],trajectory2[:,1],trajectory2[:,2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Rossler')


    ax2 = fig.add_subplot(2,1,2)
    ax2.hist(lambdas[0],
             weights = lambdas[1],
             bins = 25,
             color = 'xkcd:blue',
             label = r'$\lambda_i$, mean=' + f'{lyapunov_lambda:.04}')

    ax3 = ax2.twinx()
    ax3.hist(lambdas[1],
             bins = 25,
             color = 'xkcd:red',
             label = 'T')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')

    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
