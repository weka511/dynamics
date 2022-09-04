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
from numpy                  import argmin, argsort, argwhere, array, dot, linspace, real, size, zeros
from scipy.linalg           import inv, norm
from scipy.interpolate      import splev, splprep, splrep
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform
from section                import Section
from utils                  import get_plane, Figure


class Recurrences:
    '''This class keeps track of the recurrences of the Poincare map'''
    def __init__(self,section):
        self.section   = section

    def build2D(self,crossings):
        '''
        Build up a collection of crossings organized by distance from centre,
        plus an interpolation polynomial that we can use to find fixed points.
        '''
        self.points2D               = self.section.project_to_section(array([point for _,point in crossings]))
        self.Sorted, ArcLengths, sn = self.sort_by_distance_from_centre(self.points2D)
        self.Interpolated           = self.build_interpolated( ArcLengths, sn)
        sn1                         = sn[0:-1]
        sn2                         = sn[1:]
        isort                       = argsort(sn1)
        self.sn1                    = sn1[isort]
        self.sn2                    = sn2[isort]
        self.tckReturn              = splrep(self.sn1,self.sn2)
        self.snPlus1                = splev(self.sArray, self.tckReturn)

    def sort_by_distance_from_centre(self,points2D):
        '''
        Organize crossings by distance from centre.

        Returns:
             Sorted       Crossings ordered by distance from centre. Coordinates are relative to Section
             ArcLengths   arclengths of the Poincare section points ordered by distance from centre
             sn           arclengths of the Poincare section points in dynamical order
        '''
        Distance   = squareform(pdist(points2D))
        Sorted     = points2D.copy()
        n          = size(Sorted,0)
        ArcLengths = zeros(n)
        sn         = zeros(n)
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

            ArcLengths[k + 1]                                = ArcLengths[k] + Distance[k, k + 1]
            sn[argwhere(points2D[:, 0] == Sorted[k + 1, 0])] = ArcLengths[k + 1]

        return  Sorted, ArcLengths, sn

    def build_interpolated(self, ArcLengths, sn, num  = 1000):
        '''
        Represent arclengths by an interpolation
        '''
        self.tckPoincare,_ = splprep([self.Sorted[:, 0], self.Sorted[:, 1]],
                                     u = ArcLengths,
                                     s = 0)
        self.sArray       = linspace(min(ArcLengths), max(ArcLengths),
                                     num = num)
        return self.fPoincare(self.sArray)


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

    def ReturnMap(self,r):
        '''This function is zero when r is a fixed point of the interpolated return map'''
        return splev(r, self.tckReturn) - r

    def get_fixed(self, s0=5):
        '''Find fixed points of return map'''
        sfixed  = fsolve(self.ReturnMap, s0)[0]
        psFixed = self.fPoincare(sfixed)
        return  sfixed,psFixed,self.section.project_to_space(psFixed)



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

    return parser.parse_args()


if __name__=='__main__':
    args        = parse_args()
    section     = Section(sspTemplate = args.sspTemplate,
                          nTemplate   = args.nTemplate)
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]

    orbit       = Orbit(dynamics,
                        dt          = args.dt,
                        origin      = fp,
                        direction   = real(v),
                        eigenvalue  = w,
                        events      = section.establish_crossings())

    recurrences = Recurrences(section)
    recurrences.build2D(orbit.get_events())


    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 12,
                height   = 12) as fig:

        fig.suptitle(dynamics.get_title())
        ax   = fig.add_subplot(2,2,1,projection='3d')
        xyz  = section.get_plane(orbit)
        crossings = array([ssp for _,ssp in orbit.get_events()])
        ax.plot_surface(xyz[0,:], xyz[1,:], xyz[2,:],
                        color = 'xkcd:blue',
                        alpha = 0.5)
        ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                color = 'xkcd:green',
                label = f'{dynamics.name}')

        ax.scatter(crossings[:,0],crossings[:,1],crossings[:,2],
                   color = 'xkcd:red',
                   s     = 1,
                   label = 'Crossings')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax = fig.add_subplot(2,2,2)
        ax.scatter(recurrences.points2D[:, 0], recurrences.points2D[:, 1],
                   c      = 'xkcd:red',
                   marker = 'x',
                   s      = 25,
                   label  = 'Poincare Section')
        ax.scatter(recurrences.Sorted[:, 0], recurrences.Sorted[:, 1],
                c      = 'xkcd:blue',
                marker = '+',
                s      = 25,
                label  = 'Sorted Poincare Section')
        ax.scatter(recurrences.Interpolated[:, 0], recurrences.Interpolated[:, 1],
                c      = 'xkcd:green',
                marker = 'o',
                s      = 1,
                label  = 'Interpolated Poincare Section')

        ax.set_xlabel('$\\hat{x}\'$')
        ax.set_ylabel('$z$')
        ax.legend()

        ax = fig.add_subplot(2,2,3)
        ax.scatter(recurrences.sn1, recurrences.sn2,
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = 'Sorted')
        ax.scatter(recurrences.sArray, recurrences.snPlus1,
                   color  = 'xkcd:blue',
                   marker = 'o',
                   s      = 1,
                   label = 'Interpolated')
        ax.plot(recurrences.sArray, recurrences.sArray,
                color     = 'xkcd:black',
                linestyle = 'dotted',
                label     = '$y=x$')
        ax.legend()
        ax.set_title('Arc Lengths')



        fig.tight_layout()

    show()
