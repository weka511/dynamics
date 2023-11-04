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

'''Exercise 4.3.6'''

from argparse import ArgumentParser
from os.path import  basename,splitext
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from solver import rk4

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mus',
                        nargs = '*',
                        default= [-1.2, -1, -0.1, 0, 0.1, 1, 2, 2.1])
    parser.add_argument('--show',
                        default=False,
                        action='store_true')
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

def f(theta,mu):
    return mu + np.sin(theta) + np.cos(2*theta)

def plot_f(thetas,mu,ax=None,f=f):
    '''
    Plot RHS of differential equation
    '''
    y = f(thetas,mu)
    ax.plot(thetas,y,c='xkcd:blue')
    ax.set_xlim(0,2*np.pi)
    ax.set_title('$\mu=$' f'{mu}')
    ax.axhline(0,c='xkcd:red',linestyle=':')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    fixed_unstable = []
    fixed_stable = []
    for i in range(1,len(y)):
        if y[i-1] <= 0 and 0 <= y[i]:
            fixed_unstable.append(thetas[i])
        if y[i-1] >= 0 and 0 >= y[i]:
            fixed_stable.append(thetas[i])
    return np.array(fixed_unstable),np.array(fixed_stable)

def plot_fixed(fixed,ax = None,epsilon=0.5,c='xkcd:red'):
    '''
    Plot fixed points
    '''
    ylim0,ylim1 = ax.get_ylim()
    for y0 in fixed:
        for i in range(10):
            y1 = y0 - 2*np.pi * i
            if ylim0 - epsilon < y1 and y1 < ylim1 + epsilon:
                ax.axhline(y1,c=c,linestyle='dashed' if i==0 else'dotted')

def plot_solution(mu,
                  T = 4 * np.pi,
                  ax = None,
                  N = 1000,
                  n = 31,
                  rng = np.random.default_rng(),
                  fixed_unstable = None,
                  fixed_stable = None):
    '''
    Solve ODE and plot solution
    '''
    start = 4* np.pi * (rng.random(n)-0.5)
    h = T/N
    ts = np.arange(0,T,h)
    for i in range(n):
        y = np.zeros((N))
        y[0] = start[i]

        for i in range(N-1):
            y[i+1] = rk4(h,y[i],lambda theta:f(theta,mu))
        ax.plot(ts,y)
    plot_fixed(fixed_unstable,ax=ax)
    plot_fixed(fixed_stable,c='xkcd:blue',ax=ax)

    ax.set_xlim(0,T)
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\theta$')

if __name__=='__main__':
    start = time()
    args = parse_args()
    thetas = np.arange(0,2*np.pi,0.1)

    for i,mu in enumerate(args.mus):
        fig = figure(figsize = (10,10),
                     layout = 'constrained')
        fixed_unstable,fixed_stable = plot_f(thetas,mu,
                                             ax = fig.add_subplot(2,1,1))
        plot_solution(mu,
                      ax = fig.add_subplot(2,1,2),
                      fixed_unstable = fixed_unstable,fixed_stable=fixed_stable)
        fig.suptitle(__doc__)
        fig.savefig(get_name_for_save(extra=i+1))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
