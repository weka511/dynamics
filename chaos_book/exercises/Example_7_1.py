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

'''Example 7-1 Rössler attractor fixed points'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure, show
from scipy.interpolate import splrep, splev
from scipy.optimize import fsolve, minimize
from scipy.signal import argrelmin
from rossler import Rossler, create_jacobian
from solver import rk4

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    N =100000
    N1 = 1000
    a = 0.2
    b = 0.2
    c = 5.0
    delta_t = 0.01
    theta = 120
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots [False]')
    parser.add_argument('--N', default=N, type=int, help = f'Number of iterations for Orbit [{N:,}]')
    parser.add_argument('--N1', default=N1, type=int, help = f'Number of iterations for fixed point Orbit [{N1:,}]')
    parser.add_argument('--a', default= a, type=float, help = f'Parameter for Roessler equation [{a}]')
    parser.add_argument('--b', default= b, type=float, help = f'Parameter for Roessler equation [{b}]')
    parser.add_argument('--c', default= c, type=float, help = f'Parameter for Roessler equation [{c}]')
    parser.add_argument('--delta_t', default= delta_t, type=float, help=f'Stepsize for integrating Orbit [{delta_t}]')
    parser.add_argument('--theta', default = theta, type=int, help=f'Angle in degrees for Poincare Section [{theta}]')
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

class Template:
    '''Represents a flat Poincaré Section'''
    @staticmethod
    def create(thetaPoincare = np.pi/4):
        e_x         = np.array([1, 0, 0], float)  # Unit vector in x-direction
        sspTemplate = np.dot(Template.zRotation(thetaPoincare), e_x)  #Template vector to define the Poincare section hyperplane
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

def get_index_zero_crossings(orientation):
    '''
    Locate the positions where the orientation changes from -ve to +ve

    Parameters:
        orientation
    '''
    signs = np.sign(orientation)
    diffs = np.diff(signs)
    return np.where(diffs>0)[0]

def get_intersections(orientation,Orbit):
    '''
    Find points where Orbit intersects Poincaré section

    Parameters:
        orientation
        Orbit
    '''
    _,d = Orbit.shape
    index_zero_crossings = get_index_zero_crossings(orientation)
    m, = index_zero_crossings.shape
    intersections = np.empty((m,d))
    for i in range(m):
        a0 = np.linalg.norm(Orbit[index_zero_crossings[i],:])
        a1 = np.linalg.norm(Orbit[index_zero_crossings[i]+1,:])
        intersections[i,:] = (a1*Orbit[index_zero_crossings[i],:] + a0*Orbit[index_zero_crossings[i]+1,:])/(a0+a1)
    return intersections

def create_radial(PoincareSection, seq=0):
    radii1 = PoincareSection[:-1, seq]
    radii2 = PoincareSection[1:, seq]
    isort = np.argsort(radii1)
    return radii1[isort], radii2[isort]

def fPoincare(s,tckPoincare):
    '''
    Parametric interpolation to the Poincare section

    Parameters:
        s           Arc length which parametrizes the curve, a float or dx1-dim numpy array
        tckPoincare
    Returns:
        xy = x and y coordinates on the Poincare section, 2-dim numpy array
             or (dx2)-dim numpy array
    '''
    interpolation = splev(s, tckPoincare)
    return np.array([interpolation[0], interpolation[1]], float).transpose()

def get_fp(radii1,radii2):
    '''
    '''
    r10 = radii1.min()
    r20 = radii2.min()
    r11 = radii1.max()
    r21 = radii2.max()
    tck = splrep(radii1,radii2)
    rfixed = fsolve(lambda r: splev(r, tck) - r, r11)[0]
    return rfixed, r10,r20,r11,r21

def create_orbit(sspfixed,N=1000,delta_t=0.01,dynamics=None):
    '''
    Solve equations to computer orbit

    Parameters:
        sspfixed
        N
        delta_t
        dynamics
    '''
    Orbit = np.empty((N,dynamics.d))
    Orbit[0,:] = sspfixed
    for i in range(1,N):
        Orbit[i,:] = rk4(args.delta_t,Orbit[i-1],dynamics.Velocity)
    return Orbit

def get_T1(Orbit1,N=1000,delta_t=0.01,irange=12):
    '''
    Compute time to traverse orbit once. Traversal is defined as
    minimizing the distance from the start point.
    '''
    distances = np.empty((N))
    for i in range(N):
        distances[i] = np.linalg.norm(sspfixed - Orbit1[i,:])
    # Find provisional minimum: closest point in orbit
    index_min, = argrelmin(distances)
    index_min = index_min[0]
    indices_nearby = range(index_min-irange,index_min+irange)
    # Now fit a curve and find its minimum
    fun = splrep(delta_t * np.array(indices_nearby), np.array([distances[i] for i in indices_nearby]))
    res = minimize(lambda t: splev(t,fun),delta_t*index_min)
    if res.success:
        return res.x[0],index_min
    else:
        raise Exception('Could not find minimum')

def get_stability(Jacobian,T1):
    Floquet, _ = np.linalg.eig(Jacobian[-1,:,:])
    JJ = np.dot(np.transpose(Jacobian[-1,:,:]),Jacobian[-1,:,:])
    Stretches2,_ = np.linalg.eig(JJ)
    Lyapunov = np.log(np.sqrt(Stretches2))/T1 # see (6.4) and (6.9)
    return Floquet,Lyapunov

if __name__=='__main__':
    start  = time()
    args = parse_args()

    rossler = Rossler(a = args.a, b = args.b, c = args.c)
    Orbit = create_orbit(np.zeros((3)),N=args.N,delta_t=args.delta_t,dynamics=rossler)
    template = Template.create(thetaPoincare=np.deg2rad(args.theta))
    orientation = template.get_orientation(Orbit)
    intersections = get_intersections(orientation,Orbit)
    projection = template.get_projection(intersections)
    radii1,radii2 = create_radial(projection)

    rfixed, r10,r20,r11,r21 = get_fp(radii1,radii2)
    rlims = [min(r10,r20),max(r11,r21)]
    zs1,zs2 = create_radial(projection,seq=1)
    zfixed, z10,z20,z11,z21 = get_fp(zs1,zs2)
    zlims = [min(z10,z20),max(z11,z21)]

    sspfixed = template.get_projectionT(np.array([rfixed,zfixed,0]))
    Orbit1 = create_orbit(sspfixed, N=args.N1, delta_t=args.delta_t,dynamics=rossler)

    T1,N_T1 = get_T1(Orbit1,N=args.N1,delta_t=args.delta_t)
    Jacobian,_ = create_jacobian(sspfixed, N_T1, T1/N_T1, rossler)
    Floquet,Lyapunov = get_stability(Jacobian,T1)

    fig = figure(figsize=(12,12))

    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax1.scatter(Orbit[:,0], Orbit[:,1], Orbit[:,2],
                c = np.sign(orientation),
                s = 1,
                cmap = 'bwr')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(fr'Orbit N={args.N:,}, $\delta T=${args.delta_t}')
    ax1.legend(loc = 'upper left',
               title = 'Orientation',
               handles = [mpatches.Patch(color = 'xkcd:blue',
                                         label = 'Negative'),
                          mpatches.Patch(color = 'xkcd:red',
                                         label = 'Positive')])

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
    ax3.scatter(radii1,radii2,s=1,label='$r_{n+1}$ vs. $r_n$',c='xkcd:blue')
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

    fig.suptitle('Rössler Attractor')
    fig.tight_layout(pad = 2, h_pad = 5, w_pad = 1)
    fig.savefig(get_name_for_save(extra=1))

    bbox = dict(facecolor = 'xkcd:ivory',
                edgecolor = 'xkcd:brown',
                boxstyle = 'round,pad=1')
    fig = figure(figsize=(12,12))
    ax5 = fig.add_subplot(1,2,1,projection='3d')
    ax5.scatter(sspfixed[0], sspfixed[1],sspfixed[2],
                c = 'xkcd:terracotta',
                s = 50,
                marker = 'x',
                label = 'Start')
    ax5.text(sspfixed[0], sspfixed[1],sspfixed[2],  f'({sspfixed[0]:.4f},{sspfixed[1]:.4f},{sspfixed[2]:.4f})',
             size = 12,
             zorder = 1,
             color='xkcd:terracotta',
             bbox = bbox)
    ax5.scatter(Orbit1[:,0], Orbit1[:,1], Orbit1[:,2],
                c = 'xkcd:green',
                s = 1,
                label = 'Cycle')

    ax5.text2D(0.05, 0.75,
               '\n'.join([
                   fr'T1 = {T1:.4f},$',
                   fr'$\Lambda_1=${Floquet[0]:.4e}',
                   fr'$\Lambda_2=${Floquet[1]:.4e}',
                   fr'$\Lambda_3=${Floquet[2]:.4e}',
                   fr'$\lambda_1=${Lyapunov[0]:.4e}',
                   fr'$\lambda_2=${Lyapunov[1]:.4e}',
                   fr'$\lambda_3=${Lyapunov[2]:.4e}'
                ]),
               transform=ax5.transAxes,
               bbox = bbox)
    ax5.set_title('Provisional Cycle')
    ax5.legend(loc='lower left')
    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])
    ax5.zaxis.set_ticklabels([])

    ax6 = fig.add_subplot(1,2,2,projection='3d')
    fig.suptitle('Rössler Attractor')
    fig.savefig(get_name_for_save(extra=2))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
