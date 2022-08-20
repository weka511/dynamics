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

from eqs               import Equilibrium
from Lorentz           import Lorentz
from matplotlib.pyplot import figure, plot, show, tight_layout
from numpy             import arange, array
from scipy.integrate   import solve_ivp


def get_orbit(dt          = 100.0,
              nstp        = 10000,
              epsilon     = 0.0001,
              direction   = array([1,1,1]),
              orientation = +1,
              origin      = array([0,0,0])):

    return solve_ivp(dynamics.Velocity,  (0.0, dt), origin.eq + orientation*epsilon*direction,
                         method = 'RK45',
                         t_eval = arange(0.0, dt, dt/nstp)).y


if __name__=='__main__':
    dynamics    = Lorentz()
    eqs         = Equilibrium.create(dynamics)

    orbits = [get_orbit(origin      = eqs[0],
                        direction   = v,
                        orientation = orientation) for w,v in eqs[0].get_eigendirections() for orientation in [-1,+1]]
    fig = figure(figsize=(12,12))
    for i,orbit in enumerate(orbits):
        ax  = fig.add_subplot(2,3,i+1, projection='3d')
        ax.plot(orbit[0,:],orbit[1,:],orbit[2,:],
                c          = 'xkcd:blue',
                markersize = 1,
                label      = 'Orbit')
    show()
