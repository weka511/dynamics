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

'''Verify that all integration methods give same results for initial orbit'''

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import real
from utils                  import Figure

linestyles = [ # Used to distinguish individual plots
    (0,(3,5)),
    (0,(3,6)),
    (0,(2,7)),
    (0,(2,8)),
    (0,(1,9)),
    (0,(1,10)),
]


def parse_args():
    '''Parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = 'ProtoLorentz',
                        help    = 'The Dynamics to be investigated')
    parser.add_argument('--dt',
                        type    = float,
                        default = 20.0,
                        help    = 'Time interval for integration')
    parser.add_argument('--fp',
                        type     = int,
                        default  = 1,
                        help    = 'Fixed point to start from')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Folder to store figures')

    return parser.parse_args()

if __name__=='__main__':
    args        = parse_args()
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]

    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 12,
                height   = 12) as fig:
        fig.suptitle(dynamics.get_title())
        ax   = fig.add_subplot(1,1,1,projection='3d')


        for i,method in enumerate(['RK45','RK23','DOP853','Radau','BDF','LSODA']):
            orbit       = Orbit(dynamics,
                                dt          = args.dt,
                                origin      = fp,
                                direction   = real(v),
                                eigenvalue  = w)

            ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                    linestyle = linestyles[i],
                label = f'{method}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    show()
