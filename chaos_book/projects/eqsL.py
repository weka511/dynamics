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

'''Plot orbits starting from equilibria'''

from argparse          import ArgumentParser
from dynamics          import Equilibrium, DynamicsFactory
from matplotlib.pyplot import figure, plot, show, suptitle, tight_layout
from numpy             import arange, array, real, set_printoptions
from scipy.integrate   import solve_ivp

class Orbit:
    def __init__(self,
                 dt          = 10.0,
                 nstp        = 10000,
                 epsilon     = 0.00001,
                 direction   = array([1,1,1]),
                 orientation = +1,
                 origin      = array([0,0,0]),
                 eigenvalue  = 1):
        self.orbit = solve_ivp(dynamics.Velocity,  (0.0, dt), origin.eq + orientation*epsilon*direction,
                               method = 'RK45',
                               t_eval = arange(0.0, dt, dt/nstp)).y
        self.direction   = direction
        self.eigenvalue  = eigenvalue
        self.orientation = orientation


if __name__=='__main__':
    set_printoptions(precision=2)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = DynamicsFactory.products[0])
    parser.add_argument('--dt',
                        type    = float,
                        default = 5.0)
    parser.add_argument('--orientation',
                       type     = int,
                       nargs    = '+',
                       default  = [-1,1])
    args     = parser.parse_args()
    dynamics = DynamicsFactory.create(args)
    for eq in Equilibrium.create(dynamics):
        orbits = [Orbit(dt          = args.dt,
                        origin      = eq,
                        direction   = real(v),
                        eigenvalue  = w,
                        orientation = orientation) for w,v in eq.get_eigendirections() for orientation in args.orientation]
        fig = figure(figsize=(16,16))
        for i,orbit in enumerate(orbits):
            if i==len(args.orientation):
                ax.legend()
            if i%len(args.orientation)==0:
                ax = fig.add_subplot(2,2,i//len(args.orientation)+1, projection='3d')
                ax.set_title(f'{orbit.eigenvalue:.2f} {orbit.direction}')
            ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                    c          = 'xkcd:blue' if i%2==0 else 'xkcd:red',
                    markersize = 1,
                    label      = f'{"+"if orbit.orientation>0 else "-"}')
            ax.scatter(orbit.orbit[0,0],orbit.orbit[1,0],orbit.orbit[2,0],
                       s      = 25,
                       color  = 'xkcd:black',
                       marker = 'X',
                       label  = 'Start' if i==0 else None)
            ax.scatter(orbit.orbit[0,-1],orbit.orbit[1,-1],orbit.orbit[2,-1],
                       s      = 25,
                       color  = 'xkcd:blue' if i%2==0 else 'xkcd:red',
                       marker = '+')
        suptitle(f'Orbits starting {eq.eq}')

    show()
