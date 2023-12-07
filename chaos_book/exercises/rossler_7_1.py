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
    def __init__(self,x,n):
        self.x = x
        self.n = n

    def get_orientation(self,x):
        return np.dot(x-self.x,self.n)

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
                  [0,          0,           1]],
                 float)

if __name__=='__main__':
    start  = time()
    args = parse_args()

    rossler = Rossler(a = args.a, b = args.b, c = args.c)
    Orbit = np.empty((args.N,rossler.d))
    Orbit[0,:] = np.zeros((3))
    for i in range(1,args.N):
        Orbit[i,:] = rk4(args.delta_t,Orbit[i-1],rossler.Velocity)

    thetaPoincare = np.pi/4
    e_x         = np.array([1, 0, 0], float)  # Unit vector in x-direction
    sspTemplate = np.dot(zRotation(thetaPoincare), e_x)  #Template vector to define the Poincare section hyperplane
    nTemplate   = np.dot(zRotation(np.pi/2), sspTemplate)  #Normal to this plane will be equal to template vector rotated pi/2 about the z axis

    template = Template(sspTemplate,nTemplate)
    orientation = template.get_orientation(Orbit)
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.scatter(Orbit[:,0], Orbit[:,1], Orbit[:,2],c=orientation,s=1,cmap='seismic')

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
