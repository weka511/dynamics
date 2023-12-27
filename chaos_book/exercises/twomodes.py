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
    parser.add_argument('action', choices=['solution', 'fp'])
    parser.add_argument('--T', type=float, default=10000)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--axes', type=int, nargs=3, default=[0, 2, 3])
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--symmetry', choices = ['reduce','ignore'], default = 'ignore')
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
        [(mu1 - r2)*x1 + c1*(x1*x2 + y1*y2),
         (mu1 - r2)*y1 + c1*(x1*y2 - x2*y1),
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

def groupTransform(state, phi):
    '''
    perform group transform on a particular state. Symmetry group is 'g(phi)'
    and state is 'x'. the transformed state is ' xp = g(phi) * x '

    state:  state in the full state space. Dimension [1 x 4]
    phi:    group angle. in range [0, 2*pi]
    return: the transformed state. Dimension [1 x 4]
    '''
    c1 = np.cos(phi)
    s1 = np.sin(phi)
    c2 = np.cos(2*phi)
    s2 = np.sin(2*phi)
    return np.dot(np.array(
        [[c1,  -s1, 0,   0],
         [s1,  c1,  0,   0],
         [0,   0,   c2,  -s2],
         [0,   0,   s2,  c2]]),
                  state)

def reduceSymmetry(states,
                   show_phi = False,
                   epsilon  = 1e-15):
    '''
    transform states in the full state space into the slice.
    Hint: use numpy.arctan2(y,x)
    Note: this function should be able to reduce the symmetry
    of a single state and that of a sequence of states.

    states: states in the full state space. dimension [m x 4]
    return: the corresponding states on the slice dimension [m x 3]
    '''
    if states.ndim == 1: # if the state is one point
        phi = - np.arctan2(states[1],states[0])
        reducedStates = groupTransform(states, phi)
        assert np.abs(reducedStates[1])<epsilon
        reducedStates = np.array([reducedStates[i] for i in [0,2,3]])
        if show_phi: return reducedStates,phi
    if states.ndim == 2: # if they are a sequence of state points
        reducedStates = np.empty((3,states.shape[1]))
        for i in range(states.shape[1]):
            reducedStates[:,i] = reduceSymmetry(states[:,i])

    return reducedStates

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
                    match args.symmetry:
                        case 'reduce':
                            y = reduceSymmetry(solution.y)
                        case 'ignore':
                            y = solution.y[[args.axes[0],args.axes[1],args.axes[2]],:]
                    ax3.scatter(y[0,:],y[1,:],y[2,:],
                                s = 1,
                                c = colours[j])
                    ax3.scatter(y[0,0],y[1,0],y[2,0],
                                marker = 'X',
                                label = 'Start',
                                c = colours[j])
                ax3.set_title( Itineraries[i])
                ax3.set_xlabel(labels[args.axes[0]])
                ax3.set_ylabel(labels[args.axes[1]])
                ax3.set_zlabel(labels[args.axes[2]])
            fig.suptitle(f'{args.symmetry.title()} Symmetry')

    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
