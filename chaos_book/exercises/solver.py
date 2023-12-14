#!/usr/bin/env python

# Copyright (C) 2014-2023 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

'''Runge Kutta using numpy'''

from abc import ABC, abstractmethod
import numpy as np
from matplotlib.pyplot import figure, show

class Solver(ABC):
    @abstractmethod
    def solve(self,h,y,f=lambda y:y):
        ...

class RK4(Solver):

    def solve(self,h,y,f=lambda y:y):
        '''
        Traditional 4th order Runge Kutta

            Parameters:
                h      Step size
                y      Initial value for y in y'=f(y)
                f     Function in y=f(y)
        '''
        k1 = h * f(y)
        k2 = h * f(y + 0.5*k1)
        k3 = h * f(y + 0.5*k2)
        k4 = h * f(y + k3)
        return y + (k1 + 2*k2 + 2*k3 + k4)/6

class KuttaMerson(Solver):
    '''
    Kutta-Merson method

    https://encyclopediaofmath.org/wiki/Kutta-Merson_method

    '''
    def __init__(self,tol=1.0e-12,N=6):
        self.depth = 0
        self.tol = tol
        self.N = N

    def solve(self,h,y,f=lambda y:y):
        for i in range(self.N):
            y1 = y
            K = 2**self.depth
            h_too_small = True
            for j in range(K):
                y1,R = self.step(h/K,y1,f=f)
                if R>self.tol/64:
                    h_too_small = False
                if R>self.tol:
                    self.depth += 1
                    break

            if h_too_small  and self.depth > 0:
                self.depth -= 1
            return y1

        raise(f'Failed to converge within {tol} in {N} iterations')

    def step(self,h,y,f=lambda y:y):
        k1 = h * f(y)
        k2 = h * f(y + k1/3.0)
        k3 = h * f(y + k1/6.0 + k2/6.0)
        k4 = h * f(y + k1/8.0 + 3.0*k3/8.0)
        k5 = h * f(y + k1/2.0 - 3*k3/2.0 + 2*k4)
        y1 = y + k1/2.0 - 3.0*k3/2.0 + 2.0*k4
        y2 = y + k1/6.0 + 2.0*k4/3.0 + k5/6.0
        return y2,0.2*np.linalg.norm(y1-y2)

def Create(args):
    if args.solver == 'rk4': return RK4()
    if args.solver == 'km': return KuttaMerson(tol=args.tol)

if __name__ == '__main__':
    m = 100
    rk4 = RK4()
    y = np.zeros((m+1,2))
    y[0,1] = 1
    for i in range(1,m+1):
        y[i,:] = rk4.solve(2*np.pi/m, y[i-1,:], lambda y:np.array([-y[1],y[0]]))

    fig = figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y[:,0],y[:,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{y[-1,0]:04f},{y[-1,1]:04f},{y[-1,0]**2 +y[-1,1]**2:04f}')

    km = KuttaMerson()
    y = np.zeros((m+1,2))
    y[0,1] = 1
    for i in range(1,m+1):
        y[i,:] = km.solve(2*np.pi/m, y[i-1,:], lambda y:np.array([-y[1],y[0]]))

    fig = figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(y[:,0],y[:,1],c='xkcd:blue')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{y[-1,0]:04f},{y[-1,1]:04f},{y[-1,0]**2 +y[-1,1]**2:04f}')

    show()
