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

    solution = solve_ivp(dynamics.Velocity,  (0.0, dt), eqs[0] + sign*epsilon*(eqs[0] - eqs[1]),
                         method = 'RK45',
                         t_eval = arange(0.0, dt, dt/nstp))
    return solution.y

if __name__=='__main__':
    dynamics = Rossler()
    eqs = dynamics.find_equilibria()


    fig = figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    orbit_plus = get_orbit(eqs, dt=50,nstp=1000)
    ax.plot(orbit_plus[0,:],orbit_plus[1,:],orbit_plus[2,:],
            c          = 'xkcd:blue',
            markersize = 1)
    orbit_minus = get_orbit(eqs,
                            sign = -1,
                            dt   = 1000.0,
                            nstp = 50000,)
    ax.plot(orbit_minus[0,:],orbit_minus[1,:],orbit_minus[2,:],
            c          = 'xkcd:red',
            markersize = 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()

