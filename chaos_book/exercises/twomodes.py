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

'''Chaosbook Example 12.8: Visualize two-modes flow, Example 12.9 short relative periods, exercise 12.7'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
from xkcd import create_colour_names

mu1 = -2.8
a2 = -2.66
c1 = -7.75

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--T', type=float, default=10000)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--axes', type=int, nargs=3, default=[2,0,1])
    parser.add_argument('--action', choices=['solution', 'fp'], default='solution')
    parser.add_argument('--n', type=int, default=1)
    return parser.parse_args()

ix1 = 0
iy1 = 1
ix2 = 2
iy2 = 3
labels = ['x1','y1', 'x2','y2']

def Velocity(t,ssp):
    '''
    Velocity of two-modes system

    Parameters:
        t        Time: not used, but solve_ivp will expect it
        ssp      State space point

    Returns:
         computed velocity
    '''
    x1 = ssp[ix1]
    y1 = ssp[iy1]
    x2 = ssp[ix2]
    y2 = ssp[iy2]
    r2 = x1**2 + y1**2
    return np.array(
        [(mu1-r2)*x1 + c1*(x1*x2 + y1*y2),
         (mu1-r2)*y1 + c1*(x1*y2 - x2*y1),
         x2 + y2 + x1**2 - y1**2 + a2*x2*r2 ,
         -x2 + y2+ 2*x1*y1 + a2*y2*r2],
        dtype = float)

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
    match args.action:
        case 'solution':
            rng = np.random.default_rng()
            ssp = rng.uniform(-1,1,4)
            solution = solve_ivp(Velocity, (0,args.T), ssp)

            fig = figure(figsize=(12,12))

            ax1 = fig.add_subplot(1,1,1,projection='3d')
            ax1.scatter(solution.y[args.axes[0],:],solution.y[args.axes[1],:],solution.y[args.axes[2],:],
                        s = 1,
                        c = 'xkcd:blue',
                        label = 'Trajectory')
            ax1.scatter(solution.y[args.axes[0],0],solution.y[args.axes[1],0],solution.y[args.axes[2],0],
                        marker = 'X',
                        label = 'Start',
                        c = 'xkcd:red')

            ax1.set_xlabel(labels[args.axes[0]])
            ax1.set_ylabel(labels[args.axes[1]])
            ax1.set_zlabel(labels[args.axes[2]])
            ax1.legend()
            ax1.set_title(f'Figure 12.1 T={args.T}')

        case 'fp':
            colours = create_colour_names(args.n)
            fig = figure(figsize=(12,12))
            Itineraries = ['1', '01', '0111', '01101']

            Starts = np.array([[0.4525719, 0.0, 0.0509257, 0.0335428, 3.6415120],
                               [0.4517771, 0.0, 0.0202026, 0.0405222, 7.3459412],
                               [0.4514665, 0.0, 0.0108291, 0.0424373, 14.6795175],
                               [0.4503967, 0.0, -0.0170958, 0.0476009, 18.3874094]
                               ])
            m,_ = Starts.shape

            for i in range(m):
                ax3 = fig.add_subplot(2,2,i+1,projection='3d')
                T = Starts[i,4]
                for j in range(args.n):
                    solution = solve_ivp(Velocity, (0,T), Starts[i,0:4] if j==0 else solution.y[:,-1],
                                         t_eval=np.linspace(0,T,1000),
                                         rtol=1.0e-9,
                                         atol=1.0e-9)
                    ax3.scatter(solution.y[args.axes[0],:],solution.y[args.axes[1],:],solution.y[args.axes[2],:],
                                s = 1,
                                c = colours[j])
                    ax3.scatter(solution.y[args.axes[0],0],solution.y[args.axes[1],0],solution.y[args.axes[2],0],
                                marker = 'X',
                                label = 'Start',
                                c = colours[j])
                ax3.set_title( Itineraries[i])
                ax3.set_xlabel(labels[args.axes[0]])
                ax3.set_ylabel(labels[args.axes[1]])
                ax3.set_zlabel(labels[args.axes[2]])


    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
