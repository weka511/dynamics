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

'''Two-modes flow
   Chaosbook Example 12.8: Visualize two-modes flow,
   Example 12.9 short relative periods
   Exercise 12.7
   '''

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
    parser.add_argument('action', choices=['solution', 'fp'])
    parser.add_argument('--T', type=float, default=10000)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--axes', type=int, nargs=3, default=[0, 2, 3])
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--symmetry', choices = ['reduce','ignore', 'inslice'], default = 'ignore')
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
         -x2 + y2 + 2*x1*y1 + a2*y2*r2],
        dtype = float)

def Velocity_reduced(tau, stateVec_reduced):
    r'''
    velocity in the slice after reducing the continous symmetry

    stateVec_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    t: not used
    return: velocity at stateVect_reduced. dimension [1 x 3]
    '''
    x1 = stateVec_reduced[0]
    y1 = 0
    x2 = stateVec_reduced[1]
    y2 = stateVec_reduced[2]

    velo = Velocity(tau, [x1,y1,x2,y2])

    t = np.array([0, x1, -2*y2, 2*x2]) #Tx
    phi = Velocity_phase(stateVec_reduced)
    velo_reduced = velo - phi*t               # Equation 13.32
    return np.array([velo_reduced[i] for i in [0,2,3]])

def Velocity_phase(stateVec_reduced):
    r'''
    phase velocity.

    stateVec_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    Note: phase velocity only depends on the state vector
    '''
    x1  = stateVec_reduced[0]
    # y1         = 0
    # x2         = stateVec_reduced[1]
    y2 = stateVec_reduced[2]
    # r2         = x1**2 + y1**2
    v2 = c1*x1*y2                # (mu1-r2)*y1 + c1*(x1*y2 - x2*y1)
    return  v2/x1                  # Equation 13.33

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
    Perform group transform on a particular state. Symmetry group is 'g(phi)'
    and state is 'x'. the transformed state is ' xp = g(phi) * x '

    Parameters:
        state:  state in the full state space. Dimension [1 x 4]
        phi:    group angle. in range [0, 2*pi]
    Returns:
         the transformed state. Dimension [1 x 4]
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
                   epsilon  = 1e-15):
    '''
    transform states in the full state space into the slice.
    Note: this function should be able to reduce the symmetry
    of a single state and that of a sequence of states.

    Parameters:
        states: states in the full state space. dimension [m x 4]

    Returns:
        the corresponding states on the slice dimension [m x 3]
    '''
    if states.ndim == 1:
        phi = - np.arctan2(states[1],states[0])
        reducedStates = groupTransform(states, phi)
        assert np.abs(reducedStates[1])<epsilon
        return np.array([reducedStates[i] for i in [0,2,3]])

    if states.ndim == 2:
        reducedStates = np.empty((3,states.shape[1]))
        for i in range(states.shape[1]):
            reducedStates[:,i] = reduceSymmetry(states[:,i])
        return reducedStates

def solve_orbit(symmetry,T,Start,n=1000):
    t_eval = np.linspace(0,T,n)
    match symmetry:
        case 'reduce':
            solution = solve_ivp(Velocity, (0,T), Start,
                                 t_eval = t_eval,
                                 rtol = 1.0e-9,
                                 atol = 1.0e-9)
            return reduceSymmetry(solution.y)
        case 'ignore':
            solution = solve_ivp(Velocity, (0,T), Start,
                                 t_eval = t_eval,
                                 rtol = 1.0e-9,
                                 atol = 1.0e-9)
            return solution.y[[args.axes[0],args.axes[1],args.axes[2]],:]
        case 'inslice':
            solution = solve_ivp(Velocity_reduced, (0,T), reduceSymmetry(Start),
                                 t_eval = t_eval,
                                 rtol = 1.0e-9,
                                 atol = 1.0e-9)
            return  solution.y

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(12,12))
    match args.action:
        case 'solution':
            rng = np.random.default_rng()
            ssp = rng.uniform(-1,1,4)
            solution = solve_ivp(Velocity, (0,args.T), ssp)
            ax = fig.add_subplot(1,1,1,projection='3d')
            ax.scatter(solution.y[args.axes[0],:],solution.y[args.axes[1],:],solution.y[args.axes[2],:],
                        s = 1,
                        c = 'xkcd:blue',
                        label = 'Trajectory')
            ax.scatter(solution.y[args.axes[0],0],solution.y[args.axes[1],0],solution.y[args.axes[2],0],
                        marker = 'X',
                        label = 'Start',
                        c = 'xkcd:red')

            ax.set_xlabel(labels[args.axes[0]])
            ax.set_ylabel(labels[args.axes[1]])
            ax.set_zlabel(labels[args.axes[2]])
            ax.legend()
            ax.set_title(f'Figure 12.1 T={args.T}')

        case 'fp':
            Itineraries = ['1', '01', '0111', '01101']
            Starts = np.array([[0.4525719, 0.0, 0.0509257, 0.0335428, 3.6415120],
                               [0.4517771, 0.0, 0.0202026, 0.0405222, 7.3459412],
                               [0.4514665, 0.0, 0.0108291, 0.0424373, 14.6795175],
                               [0.4503967, 0.0, -0.0170958, 0.0476009, 18.3874094]
                               ])
            m,_ = Starts.shape
            for i in range(m):
                y = solve_orbit(args.symmetry,Starts[i,4],Starts[i,0:4])
                ax = fig.add_subplot(2,2,i+1,projection='3d')
                ax.scatter(y[0,:],y[1,:],y[2,:],
                            s = 1,
                            c = 'xkcd:blue')
                ax.scatter(y[0,0],y[1,0],y[2,0],
                            marker = 'X',
                            label = 'Start',
                            c = 'xkcd:blue')
                ax.set_title( Itineraries[i])
                ax.set_xlabel(labels[args.axes[0]])
                ax.set_ylabel(labels[args.axes[1]])
                ax.set_zlabel(labels[args.axes[2]])
            fig.suptitle(f'{args.symmetry.title()} Symmetry')

    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
