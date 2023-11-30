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

'''Hopf Bifurcations: exercises 8.2.5, 8.2.6, and 8.2.7'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from phase import generate,plot_phase_portrait,plot_stability

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--exercise', choices=['8.2.5', '8.2.6','8.2.7'], default='8.2.5')
    parser.add_argument('--mu', nargs = '+', type=float, default=[0.1])
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
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

def plot(fig = figure(figsize=(12,8)),
         f = lambda x,y,mu:(y+mu*x,-x+mu*y-x**2*y),
         suptitle = r'$\dot{x}=y + \mu x,\dot{y}=-x + \mu y - x^2 y$',
         mus = [-0.01, 0.0, 0.01],
         extra = 1):
    for i,mu in enumerate(mus):
        X,Y,U,V,fixed = generate(f = lambda x,y: f(x,y,mu),
                                 nx = 256, ny = 256,
                                 xmin = -1.0, xmax = 1.0,
                                 ymin = -1.0, ymax = 1.0)
        ax1 = fig.add_subplot(len(mus),2,2*i+1)
        plot_phase_portrait(X,Y,U,V,fixed,title = r'$\mu=$'+f'{mu}', ax = ax1)
        ax2 = fig.add_subplot(len(mus),2,2*i+2)
        plot_stability(f = lambda x,y: f(x,y,mu), R = 0.1, fixed = fixed, ax = ax2, Limit=10)

    fig.suptitle( suptitle)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(get_name_for_save(sep='_',extra=extra))

if __name__=='__main__':
    start  = time()
    args = parse_args()

    match args.exercise:
        case '8.2.5':
            plot(f = lambda x,y,mu:(y+mu*x,-x+mu*y-x**2*y),
                 suptitle = r'$\dot{x}=y + \mu x,\dot{y}=-x + \mu y - x^2 y$',
                 mus = args.mu)
        case '8.2.6':
            plot(f = lambda x,y,mu:(mu*x + y -x**3,-x+mu*y-2*y**3),
                 suptitle = r'$\dot{x}=\mu x + y -x^3 ,\dot{y}=-x + \mu y - 2 y^3$',
                 extra = 2,
                 mus = args.mu)
        case '8.2.7':
            plot(f = lambda x,y,mu:(mu*x + y -x**2,-x+mu*y-2*x**2),
                 suptitle = r'$\dot{x}=\mu x + y -x^2 ,\dot{y}=-x + \mu y - 2 x^2$',
                 extra = 3,
                 mus = args.mu)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
