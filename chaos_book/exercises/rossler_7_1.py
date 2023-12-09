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
from matplotlib.pyplot import figure, show
from scipy.interpolate import splrep, splev
from scipy.optimize import fsolve
from rossler import Rossler
from solver import rk4

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--N', default=10000, type=int)
    parser.add_argument('--a', default= 0.2, type=float)
    parser.add_argument('--b', default= 0.2, type=float)
    parser.add_argument('--c', default= 5.0, type=float)
    parser.add_argument('--delta_t', default= 0.01, type=float)
    parser.add_argument('--theta', default = 120, type=int, help='Angle in degrees')
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

def create_radial(PoincareSection):
    radii1 = PoincareSection[:-1, 0]
    radii2 = PoincareSection[1:, 0]
    isort = np.argsort(radii1)
    return radii1[isort], radii2[isort]

if __name__=='__main__':
    start  = time()
    args = parse_args()

    rossler = Rossler(a = args.a, b = args.b, c = args.c)
    Orbit = np.empty((args.N,rossler.d))
    Orbit[0,:] = np.zeros((3))
    for i in range(1,args.N):
        Orbit[i,:] = rk4(args.delta_t,Orbit[i-1],rossler.Velocity)

    template = Template.create(thetaPoincare=np.deg2rad(args.theta))
    orientation = template.get_orientation(Orbit)
    intersections = get_intersections(orientation,Orbit)
    projection = template.get_projection(intersections)
    radii1,radii2 = create_radial(projection)


    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    sc = ax1.scatter(Orbit[:,0], Orbit[:,1], Orbit[:,2],
                c = np.sign(orientation),
                s = 1,
                cmap = 'bwr')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig.colorbar(sc,ax=ax1)
    ax1.set_title(fr'Orbit N={args.N:,}, $\delta T=${args.delta_t}')

    ax2 = fig.add_subplot(2,2,2)
    ax2.scatter(projection[:,0],projection[:,1],s=1)
    ax2.set_title(fr'Crossing Poincaré section -ve to +ve: $\theta=${args.theta}'+r'$^{\circ}$')
    ax2.set_xlabel('r')
    ax2.set_ylabel('z')

    r10 = radii1.min()
    r20 = radii2.min()
    r11 = radii1.max()
    r21 = radii2.max()
    tck = splrep(radii1,radii2)
    ReturnMap = lambda r: splev(r, tck) - r
    rfixed = fsolve(ReturnMap, r11)[0]
    rlims = [min(r10,r20),max(r11,r21)]
    ax3 = fig.add_subplot(2,2,3)
    ax3.scatter(radii1,radii2,s=1,label='Return Map',c='xkcd:blue')
    ax3.plot(rlims,rlims,linestyle='--',label='$r_{n+1}=r_n$',c='xkcd:aqua')
    ax3.axvline(rfixed,ymin=min(r10,r20),ymax=rfixed,linestyle=':',
                label=f'Fixed r={rfixed}',c='xkcd:olive')
    ax3.set_title('Return map')
    ax3.set_xlabel('$r_n$')
    ax3.set_ylabel('$r_{n+1}$')
    ax3.legend()

    fig.suptitle('Rössler Attractor')
    fig.tight_layout(pad=2,h_pad=1)
    fig.savefig(get_name_for_save(extra=f'{args.theta}'))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
