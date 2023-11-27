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

from argparse import ArgumentParser, ArgumentTypeError
from os.path import  basename, splitext, join
from time import time
import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm
from matplotlib.pyplot import figure, show, Circle
from matplotlib import rc
from xkcd import generate_colour_names

def parse_args():
    '''Extract values of command line arguments'''
    def range_limited_float_type(arg,
                                 min = -1,
                                 max = +1):
        '''Type function for argparse - a float within some predefined bounds'''
        try:
            f = float(arg)
        except ValueError:
            raise ArgumentTypeError('Argument must be a floating point number')
        if f < min or f > max:
            raise ArgumentTypeError(f'Argument must be within [{min},{max}]')
        return f

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-R', '--R', type=float, default=6.0, help='Centre to Centre distance')
    parser.add_argument('-a', '--a', type=float, default=1.0, help='Radius of each Disk')
    parser.add_argument('--pad',  type=float, default=0.01, help='Used for plot p/s to pad display')
    parser.add_argument('--show', default=False, action='store_true', help='Show plots')
    parser.add_argument('--pos', type=float, nargs=2,default=[0,0])
    parser.add_argument('--p', type=range_limited_float_type, default=0)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--seed', default=42, type=int,help='Initialize random number generator')
    return parser.parse_args()

def Create_Centres(R, rtol = 1e-12):
    '''
    Create centres for a trio of Disks forming an equilateral triangle

    Parameters:
        R      Distance between centres
        rtol   Relative tolerance, used to verify that distance is correct

    Returns:
        The list of centres
    '''
    Product = R * np.array([
        [1/np.sqrt(3),0],
        [-1/(2*np.sqrt(3)),1/2],
        [-1/(2*np.sqrt(3)),-1/2]
    ])
    for i in range(len(Product)):
        assert np.isclose(norm(Product[i]-Product[(i+1)%len(Product)]),R, rtol=rtol)

    return Product

def get_next_collision(pos,velocity,Centres, a,
                       skip = None):
    '''
    Determine the next disk with which particle might collide

    Parameters:
        pos        Starting poistion of particle
        velocity   Veleocty of particle
        Centres    Centres of disks
        a          Radius of disks
        skip       Optional : do not check for collision with this disk

    Returns:
        disk   Index of the disk we collide with (meaningless if time==np.inf)
        time   Time to collision (may be infinite)
    '''
    def get_collision(Centre,omit):
        '''
        Determine time to collide with some disk (may be infinite)

        Parameters:
            Centre    Specifed centre of disk to be checked
            omit      If set to true, disk won't be checked and we assume time is infinite

        Returns:
           time   Time to collision (may be infinite)
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

    Times = [get_collision(Centres[i],skip==i) for i in range(len(Centres))]
    collision_disk = np.argmin(Times)
    return collision_disk,Times[collision_disk]

def get_distance_travelled(velocity,T):
    '''
    Find distance that particle travels in specified time

    Parameters:
        velocity   Velocity of particle
        T          Time travelled
    '''
    return  T*velocity

def get_end_point(start,velocity,T):
    '''
    Find end point after travelling from specified staing point for a specified time

    Parameters:
        start     Beginning of gravel
        velocity  Velocity of particle
        T         Time travelled
    '''
    return start + get_distance_travelled(velocity,T)


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

def get_radial_vector(pos,Centre):
    '''
    Find radius vector relative to specified centre

    Parameters:
        pos       Position of particle
        Centre    Centre of disk
    '''
    return pos - Centre

def get_reflected_velocity(radius,v, rtol=1e-12):
    '''
    Reflect the velocity by reversing the component along the radius

    Parameters:
        radius    Radial vector of potion where we collide
        v         Veloecty of particle
        rtol      Used to check that no energy has been lost during collision
    '''
    normed_radius = radius / norm(radius)
    reflected_velocity = v - 2 * np.dot(normed_radius,v) * normed_radius
    assert np.isclose(norm(v),norm(reflected_velocity),rtol=rtol)
    return reflected_velocity

def generate(pos,v,Centres,
             a = 1.0,
             first_bounce = -1):
    '''
    A generator that returns successive points in trajectory

    Parameters:
        pos          Starting position of particle
        v            Initial velocity of particle
        Centres      Centres of disks
        a            Radius of each disk
        first_bounce Used to discard trajectory if first bounce is not from specified disk

    Yields (for each collision):
        disk      Index of disk that we collide with
        T         Time to next collision
        pos       Position at start of flight
        distance  Distance travelled to collision
        radius    Radial vector from cenre of disk to collision
        v         Velocity reflected from collision
    '''
    disk,T = get_next_collision(pos,v,Centres,a)

    if first_bounce > -1 and disk != first_bounce: return

    while T<np.inf:
        distance = get_distance_travelled(v,T)
        pos1 = get_end_point(pos,v,T)
        radius = get_radial_vector(pos1,Centres[disk])
        v = get_reflected_velocity(radius,v)
        yield  disk,T,pos,distance,radius,v
        pos = pos1
        disk,T = get_next_collision(pos,v,Centres,a,skip=disk)

def get_velocity(s,p,nt=False):
    '''
    Calculate velocity after Figure 9.2

    Parameters:
        s     Arc length un units of radius (a), so it is in range(-pi,+pi)
        p     Momentum
        nt    Used in debugging to return normal and tangent
    '''
    normal = np.array([np.cos(s),np.sin(s)])
    tangent = np.array([normal[1],-normal[0]])
    v = p*tangent  - np.sqrt(1-p**2)*normal
    return (v,normal,tangent) if nt else v

def p_to_velocity(p,sgn = +1):
    '''
    Convert momentum 'p' (as defined in Chaos book) to velocity vector

    Parameters:
        p        momentum as defined in Chaos book
        sgn      Controls whther y component will be vertical or horizontal
    '''
    return np.array([p,sgn*np.sqrt(1-p**2)]) #np.array([p,sgn*np.sqrt(1-p**2)])

def draw_vector(pos,vector,
               ax = None,
               head_width = 0.1,
               head_length = 0.1,
               linestyle = '--',
               colour = 'xkcd:black'):
    '''
    A wrapper around matplotlib.pyplot.arrow so we can use vectors

    Parameters:
        pos
        vector
        ax
        head_width
        head_length
        linestyle
        color
    '''
    ax.arrow(pos[0], pos[1], vector[0],vector[1],
         head_width = head_width,
         head_length = head_length,
         linestyle = linestyle,
         color = colour)

def create_pt(s,radius=1,Centre=np.array([0,0])):
    '''
    Convert arclength 's' (as defined in Chaos book) to position vector

    Parameters:
        s          arclength as defined in Chaos book
        radius     Create the point at specified distance from Centre of disk
        Centre     Centre of disk
    '''
    return Centre + radius*np.array([np.cos(s),np.sin(s)])

def draw_disks(Centres,a=1,ax=None,pad=0,colour='xkcd:cloudy blue'):
    '''
    Draw all 3 disks

    Parameters:
        Centres   Coordinates of centre of each disk
        a         Radious of each disk
        ax        Axis for plotting
        pad       Used to pad area for display
        colour    Used to render disks
    '''
    for Centre in Centres:
        ax.add_patch(Circle(Centre,radius=args.a,fc=colour))
    ax.set_aspect('equal')
    ax.set_xlim(Centres[:,0].min() - (a+pad), Centres[:,0].max() + (a+pad))
    ax.set_ylim(Centres[:,1].min() - (a+pad), Centres[:,1].max() + (a+pad))

if __name__=='__main__':
    rc('text', usetex=True)
    start  = time()
    args = parse_args()
    Centres = Create_Centres(args.R)

    fig = figure(figsize=(16,16))

    ax1 = fig.add_subplot(2,2,1)
    draw_disks(Centres,
               a = args.a,
               ax = ax1,
               pad = args.pad)

    for disk,T,pos,distance,radius,v in generate(np.array(args.pos),
                                                 p_to_velocity(args.p),
                                                 Centres,
                                                 a = args.a):
        draw_vector(pos,distance,ax=ax1,colour='xkcd:magenta')
        position_collision = pos + distance
        ax1.text(position_collision[0],position_collision[1],f'T={T:.3f}')
        ax1.scatter(Centres[disk,0],Centres[disk,1])
        draw_vector(Centres[disk,:],radius, ax=ax1,colour='xkcd:yellow',linestyle=':')
        draw_vector(position_collision,v,linestyle='-.',ax=ax1,colour='xkcd:cyan')

    ax1.set_title(fr'p={args.p}, R/a={args.R/args.a}')

    rng = default_rng(args.seed)
    colour_name_generator = generate_colour_names()
    ax2 = fig.add_subplot(2,2,2)
    draw_disks(Centres,a=args.a,ax=ax2,pad=args.pad)
    for pt in [create_pt(s,radius=args.a + args.pad,Centre=Centres[0]) for s in 2 * np.pi * rng.random(args.N)]:
        v = p_to_velocity(2*rng.random() - 1)
        ax2.scatter(pt[0],pt[1],marker='+')
        for _,_,pos,distance_to_collision,_,_ in generate(pt,v, Centres, a = args.a, first_bounce=0):
            draw_vector(pos,distance_to_collision,
                        ax = ax2,
                        colour = colour_name_generator.__next__())
    ax2.set_title(f'{args.N} starting points')

    ax3 = fig.add_subplot(2,2,3)
    colour_name_generator = generate_colour_names()
    Centres = Create_Centres(2.5)
    draw_disks(Centres,a=1,ax=ax3,pad=0.125)
    for i in range(5):
        colour = colour_name_generator.__next__()
        s = 2*np.pi*(i - 2)/5
        p = 2*rng.random() - 1
        v,normal,tangent = get_velocity(s,p,nt=True)
        pos = create_pt(s,radius=args.a,Centre=Centres[0])
        draw_vector(pos,v,ax=ax3,colour=colour)
        draw_vector(pos,normal,ax=ax3,linestyle=':',colour=colour)
        draw_vector(pos,tangent,ax=ax3,linestyle='-.',colour=colour)
    ax3.set_title('p/s')

    fig.suptitle('Pinball simulation')
    fig.tight_layout(w_pad=0.1,h_pad=1.0)
    fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
