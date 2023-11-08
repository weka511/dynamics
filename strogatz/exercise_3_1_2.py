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

'''Exercise 3.1.2: Saddle Node bifurcations'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from bifurcations import plot_bifurcation
from solver import rk4

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def plot_f(t,r,x,ax=None):
    ax.plot(t,r - x)
    ax.axhline(0,c='xkcd:red',linestyle=':')
    ax.set_title(f'r={r}')

def create_fixed(r):
    if r > 1:
        s = np.sqrt(r**2-1)
        return [np.log(r+s),np.log(r-s)]
    elif r < 1:
        return []
    else:
        return [0]

def plot_solution(starts,r,N=1000,ax=None,a=0.5,L=10):
    h = a/N
    for start in starts:
        y = np.full((N),np.nan)
        y[0] = start
        for i in range(1,N):
            temp = rk4(h,y[i-1],lambda y:r - np.cosh(y))
            if abs(temp>L):
                break
            else:
                y[i] = temp
        ax.plot (y)
    for y0 in create_fixed(r):
        ax.axhline(y0,c='xkcd:red',linestyle='dashed')

def plot_main(fig = None):
    ax = fig.add_subplot(2,2,1)
    t = np.linspace(-2,2, 100)
    x = np.cosh(t)
    rng = np.random.default_rng()
    starts = rng.choice(t,size=13)
    for i,r in enumerate([2,1,0]):
        plot_f(t,r,x,ax = fig.add_subplot(2,3,i+1))
        plot_solution(starts,r,ax = fig.add_subplot(2,3,i+4))
    fig.suptitle(r'3.1.2 $r - \cosh{x}$')
    fig.savefig(get_name_for_save(extra=1))

if __name__=='__main__':
    start  = time()
    args = parse_args()
    plot_main(fig=figure(figsize=(20,20)))
    fig=figure(figsize=(20,20))
    plot_bifurcation(create_fixed = create_fixed,
                     r = np.concatenate((np.linspace(0.75,1,200), np.linspace(1,2,200))), # Force inclusion of r==1
                     fig = fig,
                     equation = r'$\dot{x} = r - \cosh{x}$',
                     df = lambda x,r:-np.sinh(x))
    fig.savefig(get_name_for_save(extra=2))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
