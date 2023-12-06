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

'''6.3 Rössler attractor Lyapunov Exponents'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from solver import rk4

class Rossler:
    def __init__(self,
                 a = 0.2,
                 b = 0.2,
                 c = 5.7):
        self.a = a
        self.b = b
        self.c = c

    def Velocity(self,ssp):
        '''
        Velocity function for the Rossler flow

        Inputs:
        ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]

        Outputs:
            vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
        '''

        x, y, z = ssp

        dxdt = - y - z
        dydt = x + self.a * y
        dzdt = self.b + z * (x - self.c)

        return np.array([dxdt, dydt, dzdt], float)

    def StabilityMatrix(self,ssp):
        '''
        Stability matrix for the Rossler flow

        Inputs:
            ssp: State space vector. dx1 NumPy array: ssp = [x, y, z]
        Outputs:
            A: Stability matrix evaluated at ssp. dxd NumPy array
               A[i, j] = del Velocity[i] / del ssp[j]
        '''

        x, y, z = ssp  # Read state space points


        return np.array([[0, -1,     -1],
                      [1,  self.a, 0],
                      [z,  0,      x-self.c]],
                     float)


    def JacobianVelocity(self,sspJacobian):
        '''
        Velocity function for the Jacobian integration

        Inputs:
            sspJacobian: (d+d^2)x1 dimensional state space vector including both the
                         state space itself and the tangent space


        Outputs:
            velJ = (d+d^2)x1 dimensional velocity vector
        '''

        ssp        = sspJacobian[0:3]
        J          = sspJacobian[3:].reshape((3, 3))
        velJ       = np.empty_like(sspJacobian)
        velJ[0:3]  = self.Velocity(ssp)
        velTangent = np.dot(self.StabilityMatrix(ssp), J)
        velJ[3:]   = np.reshape(velTangent, 9)
        return velJ


def create_jacobian(ssp, N, delta_t, rossler,d=3):
    '''
    Jacobian function for the trajectory started on ssp, evolved for time t

    Inputs:
        ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
        t: Integration time
    Outputs:
        J: Jacobian of trajectory f^t(ssp). dxd NumPy array
    '''
    Orbit = np.empty((N+1,d))
    Jacobian = np.empty((N+1,d,d))
    Jacobian0 = np.identity(d)
    sspJacobian0  = np.zeros(d + d ** 2)
    sspJacobian0[0:d] = ssp
    sspJacobian0[d:] = np.reshape(Jacobian0, d**2)

    Jacobian[0,:,:] = sspJacobian0[d:].reshape((d, d))
    Orbit[0,:] = ssp
    sspJacobianSolution = sspJacobian0
    for i in range(N):
        sspJacobianSolution = rk4(delta_t,sspJacobianSolution,rossler.JacobianVelocity)
        Jacobian[i+1,:,:] = sspJacobianSolution[d:].reshape((d, d))
        Orbit[i+1,:] = sspJacobianSolution[0:d]
    return Jacobian,Orbit

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--N', default = 100000, type=int)
    parser.add_argument('--N0', default = 10, type=int)
    parser.add_argument('--epsilon', default = 0.000001, type=float)
    parser.add_argument('--a', default= 0.2, type=float)
    parser.add_argument('--b', default= 0.2, type=float)
    parser.add_argument('--c', default= 5.0, type=float)
    parser.add_argument('--xtol', default= 1.0, type=float)
    parser.add_argument('--delta_t', default= 0.1, type=float)
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
    rng = np.random.default_rng(args.seed)
    x0 = rng.random(3)
    n = x0/np.linalg.norm(x0)

    rossler = Rossler(a = args.a,
                      b = args.b,
                      c = args.c)
    Jacobian,Orbit = create_jacobian(x0, args.N, args.delta_t, rossler)
    lambdas = np.empty((args.N))
    for i in range(args.N):
        Jn = np.dot(Jacobian[i+1,:,:],n)
        lambdas[i] = np.log(np.dot(Jn,Jn))/(2*(i+1))
    fig = figure(figsize=(12,12))
    ax1  = fig.add_subplot(1,2,1,projection='3d')
    ax1.plot(Orbit[:,0],Orbit[:,1],Orbit[:,2],
             c = 'xkcd:green')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'Rössler attractor: a={args.a},b={args.b},c={args.c},')

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(lambdas[args.N0:])
    ax2.set_title('Lyapunov Exponents')


    fig.suptitle(f'Lyapunov Exponents for Rössler attractor: N={args.N}, xtol={args.xtol}, '
                 + r'$\delta t=$' + f'{args.delta_t}' + ('' if args.seed==None else f', seed={args.seed}'))
    fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
