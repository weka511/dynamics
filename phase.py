# Copyright (C) 2017 Greenweaves Software Pty Ltd

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

import numpy as np, matplotlib.pyplot as plt,matplotlib.colors as colors
from scipy import optimize as opt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def get_fixed_points(f,xs,ys,tolerance_near_zero=0.001,tolerance_already_found=0.001,tolerance_root_finder=0.00001):
    '''
    Determine fixed points of differential equation
    
    Parameters:
        f                        Function - dx,dy=f(x,y)
        xs                       x values from grid
        ys                       y values from grid
        tolerance_near_zero
        tolerance_already_found
        tolerance_root_finder
    '''
    def already_found(candidate,zeroes,tol=tolerance_already_found):
        xc,yc=candidate
        for x,y in zeroes:
            if abs(x-xc)<tol and abs(y-yc)<tol:
                return True
        return False
    
    def cross(w0,w1):
        return (w0<=0 and w1>=0) or (w0>=0 and w1<=0)
    
    def near_zero(u,v,tol=tolerance_near_zero):
        return abs(u)<tol and abs(v)<tol
    
    def find_crossings():
        crossings=[]
        for x0,x1 in zip(xs[:-1],xs[1:]):
            for y0,y1 in zip(ys[:-1],ys[1:]):
                u0,v0=f(x0,y0)
                u1,v1=f(x1,y1)
                if cross(u0,u1) or cross(v0,v1):
                    crossings.append(((x0+x1)/2,(y0+y1)/2))
                elif near_zero(u0,v0):
                    crossings.append((x0,y0))
                elif near_zero(u0,v1):
                    crossings.append((x0,y1))
                elif near_zero(u1,v0):
                    crossings.append((x1,y0))
                elif near_zero(u1,v1):
                    crossings.append((x1,y1))
                
        return crossings
    
    zeroes=[]
    #  Levenberg-Marquardt gives the best results for strogatz_6_1
    #  Still having problems with exercise 6.1.3, though.
    
    for result in [opt.root(adapt(f),crossing,tol=tolerance_root_finder,method='lm') for crossing in find_crossings()]:
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
    X, Y = np.meshgrid(xs, ys)
    U,V=f(X,Y)
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

def plot_phase_portrait(X,Y,U,V,fixed,title='',suptitle=''):
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
        return f(z for zrow in Z for z in zrow)
    
    plt.pcolor(X,Y,nullclines(U,V),cmap=plt.cm.Pastel1)
    plt.streamplot(X, Y, U, V, linewidth=1)
    plt.xlim(apply2D(X,f=min),apply2D(X,f=max))
    plt.ylim(apply2D(Y,f=min),apply2D(Y,f=max))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    xs=[x for (x,_) in fixed]
    ys=[y for (_,y) in fixed]
    plt.scatter(xs,ys,marker='x',s=60,c='r')
    plt.suptitle(suptitle)
    plt.title(title)

def adapt(f):
    '''
    Adapt a 2D function so it is in the form that scipy.integrate.odeint requires, i.e.:
    ((x,y)->(dx,dy))->(([x],t)->[dx])
    The adapted function may also be used by scipy.optimize.root, as t has a default value of zero.
    
        Parameters:
            f     The function to be adapted
    '''
    def adapted(x,t=0):
        u,v=f(x[0],x[1])
        return [u]+[v]
    return adapted 

if __name__=='__main__':
    from scipy.integrate import odeint
    import utilities
    
    def f(x,y):
        return x+np.exp(-y),-y
    
    t = np.linspace(0, 25, 101)
    cs = ['r','b','g','m','c','y']
    X,Y,U,V,fixed=generate(f=lambda x,y:(x+np.exp(-y),-y),nx=256, ny = 256)

    plot_phase_portrait(X,Y,U,V,fixed,title='$\dot{x}=x+e^{-y},\dot{y}=-y$',suptitle='Example 6.1.1')
    starts=[ utilities.direct_sphere(d=2,R=10) for i in range(6)]
    for xy0,i in zip(starts,range(len(starts))):
        xy = odeint(adapt(f=f), xy0, t)
        plt.plot(xy[:,0],xy[:,1],c=cs[i%len(cs)],label='({0:.3f},{1:.3f})'.format(xy0[0],xy0[1]),linewidth=3)

        
    leg=plt.legend(loc='best')
    if leg:
        leg.draggable()
    
    plt.show()