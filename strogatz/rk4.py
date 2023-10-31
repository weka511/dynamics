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

'''Runge Kutta Library'''

def rk4(h,y,f):
    '''
    Traditional 4th order Runge Kutta

        Parameters:
            h      Step size
            y      Initial value for y in y'=f(y)
            f     Function in y=f(y)
    '''
    k0 = tuple([h*f_y for f_y in f(y)])
    k1 = tuple([h*f_y for f_y in f(tuple([y+0.5*k for y,k in zip(y,k0)]))])
    k2 = tuple([h*f_y for f_y in f(tuple([y+0.5*k for y,k in zip(y,k1)]))])
    k3 = tuple([h*f_y for f_y in f(tuple([y+k for y,k in zip(y,k2)]))])
    return tuple([yy+(_k0+2*_k1+2*_k2+_k3)/6 for yy,_k0,_k1,_k2,_k3 in zip(y,k0,k1,k2,k3)])

def adapt(f):
    '''
    Adapt a 2D function so it is in the form that rk4 requires, i.e.:
    ((x,y)->(dx,dy))->(([x])->[dx])

    Parameters:
        f     The function to be adapted
    '''
    def adapted(x):
        u,v = f(x[0],x[1])
        return [u]+[v]
    return adapted
