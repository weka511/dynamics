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

import numpy as np
from matplotlib.pyplot import figure, show

def rk4(h,y,f=lambda y:y):
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

if __name__ == '__main__':
    m = 100
    y = np.zeros((m+1,2))
    y[0,1] = 1
    for i in range(1,m+1):
        y[i,:] = rk4(2*np.pi/m, y[i-1,:], lambda y:np.array([-y[1],y[0]]))

    fig = figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y[:,0],y[:,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{y[-1,0]:04f},{y[-1,1]:04f},{y[-1,0]**2 +y[-1,1]**2:04f}')
    show()
