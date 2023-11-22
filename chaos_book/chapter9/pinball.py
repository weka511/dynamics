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
Exercise 9.1 A pinball simulator

Implement the disk to disk maps to compute a trajectory of a pinball for a given starting point,
and a given R:a = (center-to-center distance):(disk radius) ratio for a 3-disk system.
'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from numpy.linalg import norm
from matplotlib.pyplot import figure, show, Circle

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-R', '--R', type=float, default=6.0, help='Centre to Centre distance')
    parser.add_argument('-a', '--a', type=float, default=1.0, help='Radius of each Disk')
    parser.add_argument('--show', default=False, action='store_true', help='Show plots')
    parser.add_argument('--pos', type=float, nargs=2,default=[0,0])
    parser.add_argument('--theta', type=float, default=0)
    return parser.parse_args()

def Create_Centres(R, rtol=1e-12):
    '''Create the list of centres for a trio of Disks forming an equilateral triangle'''
    Centres = R * np.array([
        [1/np.sqrt(3),0],
        [-1/(2*np.sqrt(3)),1/2],
        [-1/(2*np.sqrt(3)),-1/2]
    ])
    for i in range(len(Centres)):
        assert np.isclose(norm(Centres[i]-Centres[(i+1)%len(Centres)]),R, rtol=rtol)

    return Centres

def get_next_collision(pos,velocity,Centres, a, skip=None):
    '''
    Determine the next disk with which particle might collide
    '''
    def get_next_collision(Centre,omit):
        '''
        Determine time to collide with specified disk (may be infinite)
        '''
        if omit: return np.inf

        f_k = pos[0] - Centre[0]
        g_k = pos[1] - Centre[1]
        u_0 = velocity[0]
        v_0 = velocity[1]
        b = f_k*u_0 + g_k*v_0
        if b<0:
            disc = b**2 - f_k**2 - g_k**2 + a**2
            if disc>0:
                return -b - np.sqrt(disc)
        return np.inf
    Times = [get_next_collision(Centres[i],skip==i) for i in range(len(Centres))]
    collision_disk = np.argmin(Times)
    return collision_disk,Times[collision_disk]

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

if __name__=='__main__':
    start  = time()
    args = parse_args()
    Centres = Create_Centres(args.R)
    pos = np.array(args.pos)
    v = np.array([np.cos(args.theta),np.sin(args.theta)])
    collision,T = get_next_collision(pos,v,Centres,args.a)
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    for Centre in Centres:
        ax.add_patch(Circle(Centre,radius=args.a,fc='xkcd:green'))
    ax.set_aspect('equal')
    ax.scatter(pos[0],pos[1],marker='+')
    if T<np.inf:
        ax.arrow(pos[0],pos[1],T*v[0],T*v[1],head_width=0.1, head_length=0.1)
        ax.text(pos[0]+T*v[0],pos[1]+T*v[1],f'T={T}')
    else:
        ax.arrow(pos[0],pos[1],args.R*v[0],args.R*v[1],head_width=0.1, head_length=0.1,linestyle=':')
    fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
