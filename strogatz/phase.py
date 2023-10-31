#!/usr/bin/env python

# Copyright (C) 2017-2023 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software already_foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

'''Plot phase portrait'''

import numpy as np
from matplotlib.pyplot import cm, figure, show
import matplotlib.colors as colors
from matplotlib import rc
from scipy import optimize
from utilities import direct_surface
from rk4 import rk4, adapt

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def get_fixed_points(f,xs,ys,
                     tolerance_near_zero = 0.001,
                     tolerance_already_found = 0.001,
                     tolerance_root_finder = 0.00001):
    '''
    Determine fixed points of differential equation

    Parameters:
        f                        Function - dx,dy=f(x,y)
        xs                       x values from grid
        ys                       y values from grid
        tolerance_near_zero      Used to define whther a point is "near" the origin
        tolerance_already_found  Used to define "nearness": are two roots distinct?
        tolerance_root_finder    Used by scipy.optimize.root as termination criterion.
    '''
    def already_found(candidate,zeroes):
        '''
        test a candidate fixed point to see whether we have already found it

        Parameters:
            candidate
            zeroes
        '''
        return any([abs(x-candidate[0]) < tolerance_already_found and
                    abs(y-candidate[1]) < tolerance_already_found
                    for x,y in zeroes])


    def crosses(w0,w1):
        '''
        Determine whether a value crosses between +ve and -ve

        Parameters:
            w0     One value
            w1     Some other value

        Returns: True iff w0 and w1 have opposite signs

        '''
        return (w0 <= 0 and w1 >= 0) or (w0 >= 0 and w1 <= 0)


    def is_near_zero(x,y):
        '''
        Establish whether a point is near the origin

        Parameters:
            x
            y
        '''
        return x**2 + y**2 < tolerance_near_zero**2

    def find_crossings():
        '''
        Find all position in rectangle where value is near zero
        '''
        crossings = []
        for x0,x1 in zip(xs[:-1],xs[1:]):
            for y0,y1 in zip(ys[:-1],ys[1:]):
                u0,v0=f(x0,y0)
                u1,v1=f(x1,y1)
                if crosses(u0,u1) or crosses(v0,v1):
                    crossings.append(((x0+x1)/2,(y0+y1)/2))
                elif is_near_zero(u0,v0):
                    crossings.append((x0,y0))
                elif is_near_zero(u0,v1):
                    crossings.append((x0,y1))
                elif is_near_zero(u1,v0):
                    crossings.append((x1,y0))
                elif is_near_zero(u1,v1):
                    crossings.append((x1,y1))

        return crossings

    zeroes=[]
    #  Levenberg-Marquardt gives the best results for strogatz_6_1
    #  Still having problems with exercise 6.1.3, though.

    for result in [optimize.root(adapt(f),crossing,
                                 tol = tolerance_root_finder,
                                 method = 'lm') for crossing in find_crossings()]:
        if result.success:
            if not already_found(result.x,zeroes):
                zeroes.append(result.x)

    return zeroes

def generate(f=lambda x,y:(x,y),nx=64, ny = 64,xmin=-10,xmax=10,ymin=-10,ymax=10):
    '''
     Generate a grid X,Y and the corresponding derivatives
        Parameters
            f        Function from differential equation, dx,dy=f(x,y)
            nx       Number of x steps in grid
            ny       Number of y steps in grid
            xmin     Minimum x value
            xmax     Maximum x value
            ymin     Minimum y value
            ymax     Maximum y value
        Returns:
            X,Y     Gridpoints
            U,V     The result of applying f to each gridpoint
            fixed   Fixed points of differential equation within grid
    '''
    xs = np.linspace(xmin, xmax,nx)
    ys = np.linspace(ymin, ymax, ny)
    X,Y = np.meshgrid(xs, ys)
    U,V = f(X,Y)
    return X,Y,U,V,get_fixed_points(f,xs,ys)


@np.vectorize
def nullclines(u,v):
    '''
    Used to plot nullclines. Forgets everyting except sign of u and v

    Parameters:
        u
        v
    '''
    def setnum_offset(v,offset=0):
        return offset if v<0 else offset+1
    return setnum_offset(v) if u<0 else setnum_offset(v,offset=2)

def plot_phase_portrait(X,Y,U,V,fixed,
                        title = '',
                        xlabel = '$x$',
                        ylabel = '$y$',
                        ax = None):
    '''
    Plot nullclines, stream lines, and fixed points
        Parameters:
            X,Y     Gridpoints
            U,V     TDErivatives at each gridpoint
            fixed   Fixed points of differential equation within grid
            title   Title for plotting
            xlabel  Label on X axis, for plotting
            ylabel  Label on Y axis, for plotting
            ax      Axis for figure
    '''
    def apply2D(Z, f=min):
        '''
        Apply a function to every element in a matrix
        Used to find minimum and maximum

        Parameters:
            Z     The matrix
            f     The function to be applied
        '''
        return f(z for zrow in Z for z in zrow)

    ax.pcolormesh(X,Y,nullclines(U,V),cmap=cm.Pastel1)
    ax.streamplot(X, Y, U, V, linewidth=1)
    ax.set_xlim(apply2D(X,f=min),apply2D(X,f=max))
    ax.set_ylim(apply2D(Y,f=min),apply2D(Y,f=max))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter([x for (x,_) in fixed],[y for (_,y) in fixed],marker='x',s=60,c='r')
    ax.set_title(title)



def plot_stability(f            = lambda x,y:(x,y),
                   fixed        = [(0,0)],
                   R            = 1,
                   cs           = ['xkcd:purple',
                                   'xkcd:green',
                                   'xkcd:pink',
                                   'xkcd:brown',
                                   'xkcd:teal',
                                   'xkcd:orange',
                                   'xkcd:magenta',
                                   'xkcd:yellow'],
                   linestyles   = ['-', '--', '-.', ':'],
                   Limit        = 1.0E12,
                   N            = 1000,
                   step         = 0.1,
                   S            = 1,
                   s            = 10,
                   K            = 1,
                   legend       = True,
                   accept       = lambda _:True,
                   eps          = 0.1,
                   c_stable     = 'xkcd:blue',
                   c_unstable   = 'xkcd:red',
                   ax           = None):
    '''
     Determine stability of fixed points using a Monte Carlo method

        Parameters:
            fixed       A collection of fixed points
            R           In order to probe stability, we will generate points that are near to fixed point.
                        R defines the standard deviation
            cs          List of colours to be used for plotting
            linestyles  List of linestyles to be used for plotting
            Limit       Upper bound: when we integrate, assume unstable if any coordinate exceeds Limit
            N           Number of steps to integrate differential equation
            step        Stepsize for integration
            S           Controls display of starting points: moves them out so they can be distinguished
            s           Controls size of starting points for display
            K           Controls number of iterations. For each fixed point, iterate K times for
                        each combination of a colour with a linestyle
            legend      Indicates whether legend  pointsis to be shown
            accept      Function used to constrain the offset
            eps         Used to determine whether a point is stable
            ax          Axis for plotting
            c_stable    Colour to display stable starting points
            c_unstable  Colour to display unstable starting points
    '''
    def create_offset():
        '''
        In order to probe stability, we want to generate points that are near to fixed point.
        This function is used to define an offset from the fixed point, and, optionally,
        to constrain the offset in some way
        '''
        product = tuple(R*z for z in direct_surface(d=2))
        while not accept(product):
            product = tuple(R*z for z in direct_surface(d=2))
        return product

    def evolve_trajectory(pt):
        trajectory = [tuple(x + y for x,y in zip(pt, create_offset()))]
        for j in range(N):
            (x,y) = rk4(step,trajectory[-1],adapt(f=f))
            if abs(x) < 1.5*Limit and abs(y) < 1.5*Limit:
                trajectory.append((x,y))
            else:
                break
        return np.array(trajectory)

    starts_stable = []
    starts_unstable = []

    for pt in fixed:
        for i in  range(K*len(cs)*len(linestyles)):
            trajectory = evolve_trajectory(pt)

            if np.any(abs(trajectory[-1,:]) > Limit) :
                ax.plot(trajectory[:,0],
                         trajectory[:,1],
                         c         = cs[i%len(cs)],
                         linestyle = linestyles[(i//len(cs))%len(linestyles)],
                         linewidth = 3)
                starts_unstable.append( S*(trajectory[0,:] - pt) + pt)
            else:
                if np.all(abs(trajectory[-1,:]-trajectory[0,:])<eps):
                    starts_stable.append( S*(trajectory[0,:] - pt) + pt)

    starts_stable = np.array(starts_stable).reshape(-1,2)
    starts_unstable = np.array(starts_unstable).reshape(-1,2)
    ax.scatter(starts_stable[:,0],starts_stable[:,1],
               c = c_stable,
               marker = '*',
               s = s,
               label = 'Stable')
    ax.scatter(starts_unstable[:,0],starts_unstable[:,1],
               c = c_unstable,
               marker = '+',
               s = s,
               label = 'Unstable')
    if legend:
        title='Starting points' if S==1 else f'Starting points, scaled by {S}'
        ax.legend(title=title,loc='best').set_draggable(True)

def right_upper_quadrant(pt):
    '''
    Used to test whether a point is in the right upper quadrant,
    including the axes.
    '''
    return pt[0] >= 0 and pt[1] >= 0

def strict_right_upper_quadrant(pt):
    '''
    Used to test whether a point is in the right upper quadrant,
    excluding the axes.
    '''
    return pt[0] > 0 and pt[1] > 0

if __name__=='__main__':
    X,Y,U,V,fixed = generate(f = lambda x,y:(x+np.exp(-y),-y),nx = 256, ny = 256)
    fig = figure()
    ax = fig.add_subplot(1,1,1)
    plot_phase_portrait(X,Y,U,V,fixed,title = '$\dot{x}=x+e^{-y},\dot{y}=-y$', ax = ax)
    plot_stability(f = lambda x,y:(x+np.exp(-y),-y), fixed = fixed, ax = ax)
    fig.suptitle('Example 6.1.1')
    show()
