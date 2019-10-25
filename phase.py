# Copyright (C) 2017-2019 Greenweaves Software Limited

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

# Plot phase portrait

import numpy as np, matplotlib.pyplot as plt,matplotlib.colors as colors,utilities,rk4
from scipy import optimize
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# get_fixed_points
#
# Determine fixed points of differential equation
#    
# Parameters:
#     f                        Function - dx,dy=f(x,y)
#     xs                       x values from grid
#     ys                       y values from grid
#     tolerance_near_zero      Used to define whther a point is "near" the origin
#     tolerance_already_found  Used to define "nearness": are two roots distinct?
#     tolerance_root_finder    Used by scipy.optimize.root as termination criterion.

def get_fixed_points(f,xs,ys,tolerance_near_zero=0.001,tolerance_already_found=0.001,tolerance_root_finder=0.00001):

    # already_found
    #
    # test a candidate fied point to see whether we have already found it
    #
    # Parameters:
    #        candidate
    #        zeroes
    def already_found(candidate,zeroes):
        return any([abs(x-candidate[0])<tolerance_already_found and abs(y-candidate[1])<tolerance_already_found for x,y in zeroes])

    # crosses
    #
    # Determine whether a value crosses between +ve and -ve
    #
    # Parameters:
    #     w0
    #     w1
    def crosses(w0,w1):
        return (w0<=0 and w1>=0) or (w0>=0 and w1<=0)
 
    # is_near_zero
    #
    # Establish whether a point is near the origin 
    #
    # Parameters:
    #      u
    #      v
    
    def is_near_zero(u,v):
        return abs(u)<tolerance_near_zero and abs(v)<tolerance_near_zero

    # find_crossings
    # 
    # Find all position in rectangle where value is near zero  
    
    def find_crossings():
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
    
    for result in [optimize.root(adapt(f),crossing,tol=tolerance_root_finder,method='lm') for crossing in find_crossings()]:
        if result.success:
            if not already_found(result.x,zeroes):
                zeroes.append(result.x)
            
    return zeroes

# generate
#
# Generate a grid X,Y and the corresponding derivatives
# Parameters
#     f        Function from differential equation, dx,dy=f(x,y)
#     nx       Number of x steps in grid
#     ny       Number of y steps in grid
#     xmin     Minimum x value
#     xmax     Maximum x value
#     ymin     Minimum y value
#     ymax     Maximum y value
def generate(f=lambda x,y:(x,y),nx=64, ny = 64,xmin=-10,xmax=10,ymin=-10,ymax=10):
    xs   = np.linspace(xmin, xmax,nx)
    ys   = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    U,V  = f(X,Y)
    return X,Y,U,V,get_fixed_points(f,xs,ys)

# nullclines
#
# Used to plot nullclines. Forgets everyting except sign of u and v
#    
# Parameters:
#     u
#     v
@np.vectorize
def nullclines(u,v):
    def setnum_offset(v,offset=0):
        return offset if v<0 else offset+1        
    return setnum_offset(v) if u<0 else setnum_offset(v,offset=2)

# plot_phase_portrait
#
# Plot nullclines, stream lines, and fixed points
#    
#   Parameters:
#            X
#            Y
#            U
#            V
#            fixed
#            title
#            suptitle

def plot_phase_portrait(X,Y,U,V,fixed,title='',suptitle=''):

    # apply2D
    #
    # Apply a function to every element in a matrix
    # Used to find minimum and maximum
    #
    #    Parameters:
    #        Z     The matrix
    #        f     The function to be applied
            
    def apply2D(Z,f=min):
        return f(z for zrow in Z for z in zrow)
    
    plt.pcolor(X,Y,nullclines(U,V),cmap=plt.cm.Pastel1)
    plt.streamplot(X, Y, U, V, linewidth=1)
    plt.xlim(apply2D(X,f=min),apply2D(X,f=max))
    plt.ylim(apply2D(Y,f=min),apply2D(Y,f=max))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.scatter([x for (x,_) in fixed],[y for (_,y) in fixed],marker='x',s=60,c='r')
    plt.suptitle(suptitle)
    plt.title(title)

# adapt

#  Adapt a 2D function so it is in the form that rk4.rk4 requires, i.e.:
#  ((x,y)->(dx,dy))->(([x])->[dx])
#
#    Parameters:
#        f     The function to be adapted
def adapt(f):

    def adapted(x):
        u,v=f(x[0],x[1])
        return [u]+[v]
    return adapted 

# plot_stability
#
# Determins stability of fixed points using a Monte Carlo method
#
def plot_stability(f            = lambda x,y:(x,y),
                   fixed_points = [(0,0)],
                   R            = 1,
                   cs           = ['r','b','g','m','c','y','k'],
                   linestyles   = ['-', '--', '-.', ':']):
    for fixed_point in fixed_points:
        for i in  range(len(cs)*len(linestyles)):
            offset = utilities.direct_sphere(d=2,R=R)
            xy     = [tuple(x + y for x,y in zip(fixed_point, offset))]
            for j in range(1000):
                xy.append(rk4.rk4(0.1,xy[-1],adapt(f=f)))
            plt.plot([z[0] for z in xy],
                     [z[1] for z in xy],
                     c         = cs[i%len(cs)],
                     linestyle = linestyles[i//len(cs)],
                     label     = '({0:.3f},{1:.3f})+({2:.3f},{3:.3f})'.format(fixed_point[0],fixed_point[1],offset[0],offset[1]),
                     linewidth = 3)
                
if __name__=='__main__':
    
    def f(x,y):
        return x+np.exp(-y),-y
    
    X,Y,U,V,fixed_points=generate(f=f,nx=256, ny = 256)

    plot_phase_portrait(X,Y,U,V,fixed_points,title='$\dot{x}=x+e^{-y},\dot{y}=-y$',suptitle='Example 6.1.1')
    
    plot_stability(f=f,fixed_points=fixed_points)

    plt.legend(loc='best')
    
    plt.show()