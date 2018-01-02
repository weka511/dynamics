# Copyright (C) 2014-2016 Greenweaves Software Pty Ltd

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

"""Runge Kutta Library"""

def rk4(h,y,f):
    """
    Traditional 4th order Runge Kutta
    
        Parameters:
            h      Step size
            y      Initial value for y in y'=f(y)
            f     Function in y=f(y)
    """
    k0=tuple([h*ff for ff in f(y)])
    k1=tuple([h*ff for ff in f(tuple([y+0.5*k for y,k in zip(y,k0)]))])
    k2=tuple([h*ff for ff in f(tuple([y+0.5*k for y,k in zip(y,k1)]))])
    k3=tuple([h*ff for ff in f(tuple([y+k for y,k in zip(y,k2)]))])
    return tuple([yy+(kk0+2*kk1+2*kk2+kk3)/6 for yy,kk0,kk1,kk2,kk3 in zip(y,k0,k1,k2,k3)])

