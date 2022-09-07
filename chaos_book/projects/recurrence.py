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
Periodic orbits and desymmetrization of the Lorenz flow

This file contains classes that model the recurrences of a Poincare Section
and assist in finding cycles. It includes test code for the Rossler equation.
'''

# Much of the code has been shamelessly stolen from Newton.py
# on page https://phys7224.herokuapp.com/grader/homework3

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import arange, argmin, argsort, argwhere, array, cumsum, dot, linspace, real, size, zeros
from scipy.linalg           import inv, norm
from scipy.interpolate      import splev, splprep, splrep
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform
from section                import Section
from utils                  import get_plane, Figure



class Recurrences:
    '''This class keeps track of the recurrences of the Poincare map'''
    def __init__(self,section,crossings, num=1000):
        '''
        Initialize class and sort crossings
        '''
        self.num              = num
        self.section          = section
        self.points2D         = self.section.project_to_section(array([point for _,point in crossings]))
        self.Sorted, Distance = self.sort_by_distance_from_centre()
        self.arcs             = self.Arc(self,Distance)
        self.tckPoincare,_    = splprep([self.Sorted[:, 0], self.Sorted[:, 1]],
                                        u = self.arcs.Lengths,
                                        s = 0)

    def sort_by_distance_from_centre(self):
        '''
        Organize crossings by distance from centre.

        Returns:
             Sorted       Crossings ordered by distance from centre. Coordinates are relative to Section
             Distances    Matrix of distances between crossings
        '''
        Distance = squareform(pdist(self.points2D))
        Sorted   = self.points2D.copy()
        n        = size(Distance,0)

        for k in range(n - 1):
            m                = argmin(Distance[k, k + 1:]) + k + 1

            temp             = Sorted[k + 1, :].copy()
            Sorted[k + 1, :] = Sorted[m, :]
            Sorted[m, :]     = temp

            temp               = Distance[:, k + 1].copy()
            Distance[:, k + 1] = Distance[:, m]
            Distance[:, m]     = temp

            temp               = Distance[k + 1, :].copy()
            Distance[k + 1, :] = Distance[m, :]
            Distance[m, :]     = temp

        return Sorted, Distance

    def fPoincare(self,s):
        '''
        Parametric interpolation to the Poincare section
        Inputs:
        s: Arc length which parametrizes the curve, a float or dx1-dim numpy rray
        Outputs:
        xy = x and y coordinates on the Poincare section, 2-dim numpy array or (dx2)-dim numpy array
        '''
        interpolation = splev(s, self.tckPoincare)
        return array([interpolation[0], interpolation[1]], float).transpose()

    def get_fixed(self, s0=5):
        '''Find fixed points of return map'''
        sfixed  = fsolve(self.arcs.ReturnMap, s0)[0]
        psFixed = self.fPoincare(sfixed)
        return  sfixed,psFixed,self.section.project_to_space(psFixed)

    class Arc:
        '''
        This class represents the 1D component of the Recurrences, the arc lengths
        '''
        def __init__(self, owner, Distance):
            self.owner     = owner
            n              = size(Distance,0)
            self.Lengths   = cumsum([0]+[Distance[k, k + 1] for k in range(n - 1)])  # Arc lengths ordered by distance from centre
            self.sn        = zeros(n)                                                # Arc Lengths in dynamical order
            for k in range(n - 1):
                index_dynamical_order          = argwhere(owner.points2D[:, 0] == owner.Sorted[k + 1, 0])
                self.sn[index_dynamical_order] = self.Lengths[k + 1]
            sn1            = self.sn[0:-1]                # Arc lengths in dynamical order, skipping last
            sn2            = self.sn[1:]                  # So sn1->sn2 represents mapping
            isort          = argsort(sn1)                 # We will order mapping by sn1
            self.sn1       = sn1[isort]
            self.sn2       = sn2[isort]
            self.tckReturn = splrep(self.sn1,self.sn2)

        def create_sArray(self):
            '''Array used to plot interpolated data'''
            return linspace(min(self.Lengths), max(self.Lengths), num = self.owner.num)

        def ReturnMap(self,r):
            '''This function is zero when r is a fixed point of the interpolated return map'''
            return splev(r, self.tckReturn) - r

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
                        type     = int,
                        default  = 1,
                        help    = 'Fixed point to start from')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Folder to store figures')
    parser.add_argument('--sspTemplate',
                        nargs   = 3,
                        type    = float,
                        default = [1, 0 ,0],
                        help    = 'Template point for Poincare Section')
    parser.add_argument('--nTemplate',
                        nargs   = 3,
                        type    = float,
                        default = [0, 1, 0],
                        help    = 'Normal for Poincare Section')
    parser.add_argument('--num',
                        type    = int,
                        default = 1000,
                        help    = 'Number of points when we interpolate')
    return parser.parse_args()


if __name__=='__main__':
    args        = parse_args()
    section     = Section(sspTemplate = array(args.sspTemplate),
                          nTemplate   = array(args.nTemplate))
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]

    orbit       = Orbit(dynamics,
                        dt          = args.dt,
                        origin      = fp,
                        direction   = real(v),
                        events      = section.establish_crossings())

    recurrences             = Recurrences(section,orbit.get_events(),num = args.num)
    sfixed,psfixed,sspfixed = recurrences.get_fixed(s0 = 12)
    nstp                    = 100
    epsilon                 = 1.0e-12
    sspfixed_solution       = orbit.Flow1(args.dt,sspfixed,
                                          events = section.establish_crossings(terminal=True))

    print (sspfixed, sspfixed_solution.t_events, sspfixed_solution.y_events)
    sspfixed_solution       = orbit.Flow1(args.dt,sspfixed,
                                          t_eval = arange(0.0, sspfixed_solution.t_events[0], sspfixed_solution.t_events[0]/nstp),
                                          events = section.establish_crossings(terminal=True))

    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 10,
                height   = 10) as fig:

        fig.suptitle(dynamics.get_title())
        ax1        = fig.add_subplot(2,2,1,projection='3d')
        xyz       = section.get_plane(orbit)
        crossings = array([ssp for _,ssp in orbit.get_events()])
        ax1.plot_surface(xyz[0,:], xyz[1,:], xyz[2,:],
                        color = 'xkcd:blue',
                        alpha = 0.5)
        ax1.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                color = 'xkcd:green',
                label = f'{dynamics.name}')


        ax1.scatter(crossings[:,0],crossings[:,1],crossings[:,2],
                   color = 'xkcd:red',
                   s     = 1,
                   label = 'Crossings')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        ax2 = fig.add_subplot(2,2,2)
        Interpolated = recurrences.fPoincare(recurrences.arcs.create_sArray())
        ax2.scatter(Interpolated[:,0], Interpolated[:,1],
                c      = 'xkcd:green',
                marker = 'o',
                s      = 1,
                label  = 'Interpolated Poincare Section')
        ax2.scatter(recurrences.points2D[:, 0], recurrences.points2D[:, 1],
                   c      = 'xkcd:red',
                   marker = 'x',
                   s      = 25,
                   label  = 'Crossing Poincare Section')
        ax2.scatter(recurrences.Sorted[:, 0], recurrences.Sorted[:, 1],
                c      = 'xkcd:blue',
                marker = '+',
                s      = 25,
                label  = 'Sorted Crossings')
        ax2.scatter(psfixed[0], psfixed[1],
                   c      = 'xkcd:yellow',
                   marker = 'D',
                   s      = 25,
                   label  = 'psfixed')
        ax2.set_xlabel('$\\hat{x}\'$')
        ax2.set_ylabel('$z$')
        ax2.legend()

        ax3 = fig.add_subplot(2,2,3)
        ax3.scatter(recurrences.arcs.sn1, recurrences.arcs.sn2,
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = 'As fn(previous)')
        ax3.scatter(recurrences.arcs.create_sArray(), splev(recurrences.arcs.create_sArray(), recurrences.arcs.tckReturn),
                   color  = 'xkcd:blue',
                   marker = 'o',
                   s      = 1,
                   label = 'Interpolated')
        ax3.plot(recurrences.arcs.create_sArray(), recurrences.arcs.create_sArray(),
                color     = 'xkcd:black',
                linestyle = 'dotted',
                label     = '$y=x$')
        ax3.legend()
        ax3.set_xlabel('previous')
        ax3.set_xlabel('next')
        ax3.set_title('Arc Lengths')

        ax4   = fig.add_subplot(2,2,4,projection='3d')
        ax4.plot(sspfixed_solution.y[0,:],sspfixed_solution.y[1,:],sspfixed_solution.y[2,:],
                color = 'xkcd:purple',
                label = f'sspfixed_solution')

        fig.tight_layout()

    show()
