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
import rk4

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def get_fixed_points(f,xs,ys,tolerance_near_zero=0.001,tolerance_already_found=0.001,tolerance_root_finder=0.00001):
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
        test a candidate fied point to see whether we have already found it
        Parameters:
            candidate
            zeroes
        '''
        return any([abs(x-candidate[0])<tolerance_already_found and
                            abs(y-candidate[1])<tolerance_already_found for x,y in zeroes])


    def crosses(w0,w1):
        '''
        Determine whether a value crosses between +ve and -ve

        Parameters:
            w0
            w1
        '''
        return (w0<=0 and w1>=0) or (w0>=0 and w1<=0)


    def is_near_zero(u,v):
        '''
        Establish whether a point is near the origin

        Parameters:
            u
            v
        '''
        return abs(u)<tolerance_near_zero and abs(v)<tolerance_near_zero

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

def plot_phase_portrait(X,Y,U,V,fixed,title='',xlabel='$x$',ylabel='$y$',ax=None):
    '''
    Plot nullclines, stream lines, and fixed points
        Parameters:
            X
            Y
            U
            V
            fixed
            title
            suptitle
    '''
    def apply2D(Z,f=min):
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

def adapt(f):
    '''
    Adapt a 2D function so it is in the form that rk4.rk4 requires, i.e.:
    ((x,y)->(dx,dy))->(([x])->[dx])

    Parameters:
        f     The function to be adapted
    '''
    def adapted(x):
        u,v=f(x[0],x[1])
        return [u]+[v]
    return adapted


def plot_stability(f            = lambda x,y:(x,y),
                   fixed_points = [(0,0)],
                   R            = 1,
                   cs           = ['r','b','g','m','c','y','k'],
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
                   ax = None):
    '''
     Determins stability of fixed points using a Monte Carlo method

        Parameters:
            fixed_points
            R
            cs
            linestyles
            Limit
            N
            step
            S
            s
            K
            legend
            accept
            eps
            ax
    '''
    starts0 = []
    starts1 = []
    for fixed_point in fixed_points:
        for i in  range(K*len(cs)*len(linestyles)):
            offset = tuple(R*z for z in direct_surface(d=2))
            while not accept(offset):
                offset = tuple(R*z for z in direct_surface(d=2))
            xys  = [tuple(x + y for x,y in zip(fixed_point, offset))]

            for j in range(N):
                (x,y)=rk4.rk4(0.1,xys[-1],adapt(f=f))
                if abs(x) < 1.5*Limit and abs(y) < 1.5*Limit:
                    xys.append((x,y))
                else:
                    break

            if abs(xys[-1][0]) > Limit or abs(xys[-1][1]) > Limit:
                ax.plot([z[0] for z in xys],
                         [z[1] for z in xys],
                         c         = cs[i%len(cs)],
                         linestyle = linestyles[(i//len(cs))%len(linestyles)],
                         linewidth = 3)
                starts1.append( (xys[0]))
            else:
                if abs(xys[-1][0]-xys[0][0])<eps and abs(xys[-1][1]-xys[0][1])<eps:
                    starts0.append( (xys[0]))

    ax.scatter([S*x for (x,_) in starts0],[S*y for (_,y) in starts0],c='b',marker='*',s=s,label='Stable')
    ax.scatter([S*x for (x,_) in starts1],[S*y for (_,y) in starts1],c='r',marker='+',s=s,label='Unstable')
    if legend:
        ax.legend(title='Starting points, scaled by {0:3}'.format(S),loc='best').set_draggable(True)

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
    X,Y,U,V,fixed_points = generate(f = lambda x,y:(x+np.exp(-y),-y),nx = 256, ny = 256)
    fig = figure()
    ax = fig.add_subplot(1,1,1)
    plot_phase_portrait(X,Y,U,V,fixed_points,title = '$\dot{x}=x+e^{-y},\dot{y}=-y$', ax = ax)
    plot_stability(f = lambda x,y:(x+np.exp(-y),-y), fixed_points = fixed_points, ax = ax)
    fig.suptitle('Example 6.1.1')
    show()
