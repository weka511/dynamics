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

'''
    Example 7-1 Rössler attractor fixed points

    We run a long simulation of the Rössler flow, plot a Poincaré section, as in figure 3.3,
    and extract the corresponding return map, as in figure 3.4. Display cycle, Floquet multipliers,
    and Lyapunov exponents.
'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from scipy.integrate import solve_ivp
from scipy.interpolate import splrep, splev
from scipy.optimize import fsolve
from rossler import Rossler

def parse_args( T =1000,
                a = 0.2,
                b = 0.2,
                c = 5.0,
                theta = 120,
                figs = './figs'):
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--T', default=T, type=int, help = f'Time for initial [{T:,}]')
    parser.add_argument('--a', default= a, type=float, help = f'Parameter for Roessler equation [{a}]')
    parser.add_argument('--b', default= b, type=float, help = f'Parameter for Roessler equation [{b}]')
    parser.add_argument('--c', default= c, type=float, help = f'Parameter for Roessler equation [{c}]')
    parser.add_argument('--theta', default = theta, type=int, help=f'Angle in degrees for Poincare Section [{theta}]')
    parser.add_argument('--figs', default = figs, help=f'Pathname to save figures [{figs}]')
    return parser.parse_args()

class Template:
    '''Represents a flat Poincaré Section'''
    @staticmethod
    def create(thetaPoincare = np.pi/4):
        e_x         = np.array([1, 0, 0], float)  # Unit vector in x-direction
        sspTemplate = np.dot(Template.zRotation(thetaPoincare), e_x)  #Template vector to define the Poincaré section hyperplane
        nTemplate   = np.dot(Template.zRotation(np.pi/2), sspTemplate)  #Normal to this plane will be equal to template vector rotated pi/2 about the z axis
        e_z          = np.array([0, 0, 1], float)  # Unit vector in z direction
        ProjPoincare = np.array([sspTemplate,
                                 e_z,
                                 nTemplate])
        return Template(sspTemplate,nTemplate,ProjPoincare)

    @staticmethod
    def zRotation(theta):
        '''
        Rotation matrix about z-axis
        Input:
        theta: Rotation angle (radians)
        Output:
        Rz: Rotation matrix about z-axis
        '''
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta),  0],
                         [0,             0,              1]])

    def __init__(self,sspTemplate,nTemplate,ProjPoincare):
        self.sspTemplate = sspTemplate
        self.nTemplate = nTemplate
        self.ProjPoincare = ProjPoincare

    def get_orientation(self,ssp):
        '''
        Calculate the U(x) using Chaosbook eq (3.14).

        Parameters:
            ssp   A 3 vectopr representing the point

        Returns:
            Zero if x is on the Poincaré Section;
            it will be negative on one side, posive on the other
        '''
        return np.dot(ssp - self.sspTemplate,self.nTemplate)

    def get_projection(self,ssp):
        '''
        Project a state space point onto the Poincaré Section

        Parameters:
            ssp     State Space Point
        '''
        return np.dot(self.ProjPoincare, ssp.transpose()).transpose()

    def get_projectionT(self,ssp):
        '''
        Project a state space point onto the Poincaré Section

        Parameters:
            ssp     State Space Point
        '''
        return np.dot(ssp,self.ProjPoincare)

def create_start(dynamics=None,eps=1e-6):
    '''
    Choose a starting point for an Orbit, located on the unstable manifold near a fixed point.
    '''
    eq0 = fsolve(dynamics.Velocity, np.array([0, 0, 0], float))
    Aeq0 = dynamics.StabilityMatrix(eq0)
    _, eigenVectors = np.linalg.eig(Aeq0)
    v1 = np.real(eigenVectors[:, 0])
    v1 = v1 / np.linalg.norm(v1)
    return eq0 + eps * v1

def map_component(projection, seq=0):
    '''
    Create mapping for one specified component from one point to next

    Parameters:
        projection  Projection of orbit onto Poincaré section
        seq         Used to control whether we map first or second component

    Returns:
        firsts, successors  A tuple of lists: each point in successors
                            is the successor of the corresponding term in firsts.
                            The list firsts in sorted in asnding order
    '''
    firsts = projection[:-1, seq]
    successors = projection[1:, seq]
    isort = np.argsort(firsts)
    return firsts[isort], successors[isort]

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

def get_fp(firsts,successors):
    '''
    Find fixed point for specified component
    '''
    r10 = firsts.min()
    r20 = successors.min()
    r11 = firsts.max()
    r21 = successors.max()
    tck = splrep(firsts,successors)
    return fsolve(lambda r: splev(r, tck) - r, r11)[0], r10,r20,r11,r21

if __name__=='__main__':
    start  = time()
    args = parse_args()

    rossler = Rossler(a = args.a, b = args.b, c = args.c)
    template = Template.create(thetaPoincare=np.deg2rad(args.theta))
    get_orientation = lambda t,y: template.get_orientation(y)
    get_orientation.direction = 1.0

    solution = solve_ivp(lambda t,y:rossler.Velocity(y),(0,args.T),create_start(dynamics=rossler),
                         events = [get_orientation])

    crossings = solution.y_events[0]
    projection = template.get_projection(crossings)
    component_1,component_2 = map_component(projection)

    rfixed, r10,r20,r11,r21 = get_fp(component_1,component_2)
    rlims = [min(r10,r20),max(r11,r21)]
    zs1,zs2 = map_component(projection,seq=1)
    zfixed, z10,z20,z11,z21 = get_fp(zs1,zs2)
    zlims = [min(z10,z20),max(z11,z21)]

    fig = figure(figsize=(12,12))

    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax1.scatter(solution.y[0,:], solution.y[1,:], solution.y[2,:],s=1,c='xkcd:blue',label='Rössler')
    ax1.scatter(crossings[:,0], crossings[:,1], crossings[:,2],s=25,c='xkcd:red',label='Crossing Poincaré Section')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(fr'Orbit T={args.T:,}')
    ax1.legend()

    ax2 = fig.add_subplot(2,2,2)
    ax2.scatter(projection[:,0],projection[:,1],s=1,label='Crossings',c='xkcd:blue')
    ax2.set_title(fr'Crossing Poincaré section -ve to +ve: $\theta=${args.theta}'+r'$^{\circ}$')
    ax2.axvline(rfixed,
                linestyle=':',
                label=f'Fixed r={rfixed:.06f}',c='xkcd:hot pink')
    ax2.axhline(zfixed,
                linestyle=':',
                label=f'Fixed z={zfixed:.06f}',c='xkcd:forest green')
    ax2.set_xlabel('r')
    ax2.set_ylabel('z')
    ax2.legend(loc='upper left')

    ax3 = fig.add_subplot(2,2,3)
    ax3.scatter(component_1,component_2,s=1,label='$r_{n+1}$ vs. $r_n$',c='xkcd:blue')
    ax3.plot(rlims,rlims,linestyle='--',label='$r_{n+1}=r_n$',c='xkcd:aqua')
    ax3.axvline(rfixed,ymin=min(r10,r20),ymax=rfixed,linestyle=':',
                label=f'Fixed r={rfixed:.06f}',c='xkcd:hot pink')
    ax3.set_title('Return map (r)')
    ax3.set_xlabel('$r_n$')
    ax3.set_ylabel('$r_{n+1}$')
    ax3.legend(loc='lower right')
    ax3.set_aspect('equal')

    ax4 = fig.add_subplot(2,2,4)
    ax4.scatter(zs1,zs2,s=1,label='$z_{n+1}$ vs. $z_n$',c='xkcd:blue')
    ax4.plot(zlims,zlims,linestyle='--',label='$z_{n+1}=z_n$',c='xkcd:aqua')
    ax4.axvline(zfixed,linestyle=':',
                label=f'Fixed z={zfixed:.06f}',c='xkcd:olive')
    ax4.set_title('Return map (z)')
    ax4.set_xlabel('$z_n$')
    ax4.set_ylabel('$z_{n+1}$')
    ax4.legend(loc='lower right')
    ax4.set_aspect('equal')

    fig.suptitle(f'Rössler Attractor')
    fig.tight_layout(pad = 2, h_pad = 5, w_pad = 1)
    fig.savefig(get_name_for_save(figs=args.figs, extra=1))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
