#!/usr/bin/env python

# Copyright (C) 2022 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

'''Replicate Figure 4.5'''

from Lorentz           import Rossler
from matplotlib.pyplot import figure, plot, show
from numpy             import arange
from scipy.integrate   import solve_ivp
from scipy.linalg      import eig

def get_orbit(eqs,
              dt      = 100.0,
              nstp    = 10000,
              epsilon = 0.0001,
              sign     = +1):

    return solve_ivp(dynamics.Velocity,  (0.0, dt), eqs[0] + sign*epsilon*(eqs[0] - eqs[1]),
                         method = 'RK45',
                         t_eval = arange(0.0, dt, dt/nstp)).y


if __name__=='__main__':
    dynamics    = Rossler()
    eqs         = dynamics.find_equilibria()
    orbit_plus  = get_orbit(eqs, dt=50)
    orbit_minus = get_orbit(eqs,
                            sign = -1)
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.plot(orbit_plus[0,:],orbit_plus[1,:],orbit_plus[2,:],
            c          = 'xkcd:blue',
            markersize = 1)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(orbit_minus[0,:],orbit_minus[1,:],orbit_minus[2,:],
            c          = 'xkcd:red',
            markersize = 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    show()
