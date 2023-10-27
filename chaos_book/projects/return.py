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

# This program is intended to support my project from https://chaosbook.org/
#  Chaos: Classical and Quantum - open projects

'''
Replicate calculations form Arindam Basu's paper
'''

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import arange, array, real, searchsorted,stack
from section                import Section
from sys                    import float_info
from utils                  import get_plane, Figure

def parse_args():
    '''Parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = 'Rossler',
                        help    = 'The Dynamics to be investigated')
    parser.add_argument('--dt',
                        type    = float,
                        default = 900.0,
                        help    = 'Time interval for integration')
    parser.add_argument('--fp',
                        type    = int,
                        default = 1,
                        help    = 'Fixed point to start from')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Folder to store figures')
    parser.add_argument('--N',
                        type    = int,
                        default = 100,
                        help    = 'Used to establish mapping')

    return parser.parse_args()

def get_y_cross(y0):
    _,y_start = orbit.Flow(1,y0)
    return orbit.FlowToNextCrossing(args.dt,y_start,events = section.establish_crossings(terminal=True)).y_events[-0][0,:]

if __name__ == '__main__':
    args        = parse_args()
    section     = Section(
                    sspTemplate = array([1,0,0]),
                    nTemplate   = array([0,1,0]))
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]

    orbit       = Orbit(dynamics,
                        dt          = args.dt,
                        origin      = fp,
                        direction   = real(v),
                        events      = section.establish_crossings(terminal=False))

    crossings    = orbit.get_events()
    t0,y0        = next(crossings)
    t1,y1        = next(crossings)
    delta_y      = (y1-y0)/(args.N+1)
    ys           = stack([y0+i*delta_y for i in range(args.N)])
    y_recurrence = stack([get_y_cross(ys[i,:]) for i in range(args.N)])
    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 12,
                height   = 12) as fig:

        fig.suptitle(dynamics.get_title())
        ax   = fig.add_subplot(1,1,1,projection='3d')

        section.plot_surface(ax,orbit,
                             t0 = t0,
                             t1 = t1)
        i0 = searchsorted(orbit.t,t0)
        i1 = searchsorted(orbit.t,t1)
        ax.plot(orbit.y[0,i0:i1],orbit.y[1,i0:i1],orbit.y[2,i0:i1],
                color = 'xkcd:green',
                label = f'{dynamics.name}')

        ax.scatter(y0[0], y0[1], y0[2],
                   color = 'xkcd:red',
                   s     = 25,
                   marker = 'X',
                   label = '$X_A$')
        ax.scatter(y1[0], y1[1], y1[2],
                   color = 'xkcd:red',
                   s     = 25,
                   marker = '+',
                   label = '$X_B$')

        ax.scatter(y_recurrence[:,0], y_recurrence[:,1], y_recurrence[:,2],
                   color = 'xkcd:purple',
                   s     = 1,
                   label = '$Recurrences(X_A,X_B)$')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

show()
