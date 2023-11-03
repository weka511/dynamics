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
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

def f(theta,mu):
    return mu + np.sin(theta) + np.cos(2*theta)

def plot_f(thetas,mu,ax=None):
    ax.plot(thetas,f(thetas,mu),c='xkcd:blue')
    # ax.set_xlabel(r'$\theta$')
    # ax.set_ylabel(r'$\mu +\sin {\theta} + \cos {2 \theta}$')
    ax.set_title('$\mu=$' f'{mu}')
    ax.axhline(0,c='xkcd:red',linestyle=':')

def plot_sol(mu,ax=None,N=1000,n=13,rng = np.random.default_rng(),swapped = False):
    start = 4* np.pi * (rng.random(n)-0.5)
    for i in range(n):
        y = np.zeros((N))
        y[0] = start[i]
        h = 2* np.pi/N
        for i in range(N-1):
            y[i+1] = rk4(h,y[i],lambda theta:f(theta,mu))
        ax.axhline(2*np.pi,c='xkcd:red',linestyle=':')
        ax.axhline(0,c='xkcd:red',linestyle=':')
        while swapped:
            swapped = False
            for i in range(N):
                if y[i] > 2*np.pi:
                    y[i] -= 2*np.pi
                    swapped = True
                elif y[i] < 2*0:
                        y[i] += 2*np.pi
                        swapped = True
        ax.plot(y)

if __name__=='__main__':
    start = time()
    args = parse_args()
    thetas = np.arange(0,2*np.pi,0.1)
    mus = [-1.2, -1.1, -1, -0.1, 0,0.1, 1,2,2.1]
    fig = figure(figsize=(10,20))
    for i,mu in enumerate(mus):
        plot_f(thetas,mu,
               ax=fig.add_subplot(2,len(mus),i+1))
        plot_sol(mu,
                 ax=fig.add_subplot(2,len(mus),i+len(mus)+1))
    fig.suptitle(__doc__)
    fig.tight_layout()
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
