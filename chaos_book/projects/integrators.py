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

'''
Verify that all integration methods give same results for initial orbit

Spoiler: they don't!
'''

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import real
from utils                  import Figure, Timer, xkcd_colour_names




def get_linestyle(i,n=7,m=5):
    '''Used to distinguish individual plots'''
    return (0, ((n-i)//2,m+i))

def parse_args():
    '''Parse command line arguments'''
    methods = ['RK45','RK23','DOP853','Radau','BDF','LSODA']
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = 'ProtoLorentz',
                        help    = 'The Dynamics to be investigated')
    parser.add_argument('--methods',
                        choices = methods,
                        nargs   = '+',
                        default = methods,
                        help    = 'Allows us to focus on a subset of the available methods')
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
    parser.add_argument('--linestyles',
                        default = False,
                        action = 'store_true',
                        help   = 'Use a different linestyle for each method')
    parser.add_argument('--show',
                        default = False,
                        action = 'store_true',
                        help   = 'Show plot (otherwise, just save to file)')
    return parser.parse_args()

if __name__=='__main__':
    args        = parse_args()
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]
    colour      = xkcd_colour_names()
    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 12,
                height   = 12) as fig:
        fig.suptitle(dynamics.get_title())
        ax   = fig.add_subplot(1,1,1,projection='3d')

        for i,method in enumerate(args.methods):
            with Timer(silent=True) as timer:
                orbit = Orbit(dynamics,
                              method     = method,
                              dt         = args.dt,
                              origin     = fp,
                              direction  = real(v),
                              eigenvalue = w)

                ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                        linestyle = get_linestyle(i) if args.linestyles else None,
                        c         = next(colour),
                        label     = f'{method:>8}, nfev={orbit.nfev:>6}, dt={timer.get_elapsed():>.2f} sec.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(prop={'family': 'monospace'})

    if args.show:
        show()
