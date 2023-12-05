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

'''Exercise 6.3 How unstable is the Hénon attractor?'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

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

def henon(x = 0,
          a = 1.4,
          b = 0.3):
    '''Get next point for Hénon map'''
    return np.array([1 - a*x[0]**2 + x[1], b*x[0]])

def evolve(x0,near,
           mapping = lambda x:henon(x),
           N = 100000,
           xtol = 1.0,
           delta_t = 1.0):
    m, = x0.shape
    trajectory1 = np.empty((N,m))
    trajectory1[0,:] = x0
    trajectory2 = np.empty((N,m))
    trajectory2[0,:] = x0 + near
    lyapunov = []
    lyapunov_lambda = 0
    i0 = 0
    dx0 = np.linalg.norm(near)
    for i in range(1,N):
        trajectory1[i,:] = mapping(trajectory1[i-1,:])
        trajectory2[i,:] = mapping(trajectory2[i-1,:])
        dx = np.linalg.norm(trajectory1[i,:] - trajectory2[i,:])
        if dx>xtol:
            t = delta_t * (i - i0)
            factor = dx/dx0
            lyapunov.append(( np.log(factor)/t,t))
            trajectory2[i,:] = trajectory1[i,:]  + (trajectory2[i,:]-trajectory1[i,:] )/factor
            i0 = i
            lyapunov_lambda +=  np.log(factor)
    return trajectory1,trajectory2,lyapunov_lambda/(N*delta_t),lyapunov

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
    rng = np.random.default_rng(args.seed)
    x0 = np.array([0,0])
    x1 = x0 + rng.uniform(0.0,args.epsilon,size=2)
    mapping = lambda x:henon(x,a=args.a,b=args.b)
    if args.N0==None:
        trajectory1,trajectory2,lyapunov_lambda,lyapunov = evolve(x0,x1,
                                                                  mapping = mapping,
                                                                  N = args.N,
                                                                  xtol = args.xtol)
    else:
        trajectory1,trajectory2,lyapunov_lambda,lyapunov = evolve(x0,x1,
                                                                  mapping = mapping,
                                                                  N = args.N0,
                                                                  xtol = args.xtol)
        trajectory1,trajectory2,lyapunov_lambda,lyapunov = evolve(trajectory1[-1,:],trajectory2[-1,:],
                                                                  mapping = mapping,
                                                                  N = args.N,
                                                                  xtol = args.xtol)
    lambdas = list(zip(*lyapunov))
    fig = figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1)
    ax1.scatter(trajectory1[:,0],trajectory1[:,1],
                c = 'xkcd:blue',
                s = 1,
                label = 'Unperturbed',
                alpha = 0.5)
    ax1.scatter(trajectory2[:,0],trajectory2[:,1],
                c = 'xkcd:red',
                s = 1,
                label = r'Perturbed $\epsilon=$'+f'{args.epsilon}',
                alpha = 0.5)
    ax1.set_title(f'Hénon attractor a={args.a}, b={args.b}')
    ax1.legend()
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

    suptitle = (f'N={args.N:,}, epsilon={args.epsilon}, xtol={args.xtol}' if args.N0==None
                else  f'N={args.N:,}(burn in: {args.N0:,}), epsilon={args.epsilon}, xtol={args.xtol}')
    fig.suptitle(suptitle)
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
