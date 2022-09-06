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
from recurrence             import Recurrences
from scipy.linalg           import inv, norm
from scipy.interpolate      import splev, splprep, splrep
from scipy.optimize         import fsolve
from section                import Section
from utils                  import get_plane, Figure


class CycleFinder:
    '''Used to find cycles in flow'''
    def __init__(self,
                 section     = None,
                 recurrences = None,
                 orbit      = None):
        self.recurrences = recurrences
        self.section     = section
        self.orbit       = orbit

    def find_initial_cycle(self,
              dt0   = 0,
              s0    = 0):
        '''First approximation to cycle'''
        sfixed,psfixed,sspfixed = self.recurrences.get_fixed(s0 = s0)
        Tguess                  = dt0 / size(self.recurrences.Sorted, 0)
        return fsolve(lambda t: self.section.U(self.orbit.Flow(t,sspfixed)[1]), Tguess)[0], sfixed, psfixed, sspfixed

    def refine(self, Tnext, sspfixed,
               tol        = 1e-9,
               kmax       = 20,
               orbit      = None,
               nstp       = 1000,
               freq       = None,
               alpha      = 1.0):
        '''Use Newton's method to refine an estimate for a cycle. See Chaos Book chapter 7'''
        def iterating():
            '''Iterator for Newton's mthod'''
            k  = 0
            while max(abs(error)) > tol:
                if k > kmax:
                    raise Exception("Passed the maximum number of iterations")
                k += 1
                if freq != None and k%freq == 0:
                    print (f'Iteration {k} error = {max(abs(error))}')
                yield k

        d          = orbit.dynamics.d
        period     = Tnext.copy()
        error      = zeros(d+1)
        Delta      = zeros(d+1)
        error[0:d] = orbit.Flow(period,sspfixed)[1] - sspfixed
        Newton     = zeros((d+1, d+1))

        for k in iterating():
            Newton[0:d, 0:d] = 1 - orbit.Jacobian(Tnext,sspfixed)
            Newton[0:d, d]   = - orbit.dynamics.Velocity(Tnext,sspfixed,)
            Newton[d, 0:d]   = self.section.nTemplate
            Delta            = dot(inv(Newton), error)
            sspfixed         = sspfixed +alpha* Delta[0:d]
            period           = period + alpha*Delta[d]
            error[0:d]       = orbit.Flow(period, sspfixed)[1] - sspfixed

        return period, sspfixed, orbit.Flow(period,sspfixed,nstp=nstp)[1]

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
    parser.add_argument('--s0',
                        type    = float,
                        default = 12.0)
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
                        eigenvalue  = w,
                        events      = section.establish_crossings())

    recurrences = Recurrences(section,orbit.get_events(),num = args.num)
    cycle_finder                  = CycleFinder(section     = section,
                                                recurrences = recurrences,
                                                orbit       = orbit)
    dt, sfixed, psfixed, sspfixed = cycle_finder.find_initial_cycle(s0    = args.s0,
                                                                    dt0   = args.dt)

    sspfixedSolution                = orbit.Flow(dt,sspfixed, nstp=1000)[1]
    period, sspfixed1, periodicOrbit = cycle_finder.refine(dt, sspfixed, orbit      = orbit)
    print("Shortest periodic orbit is at: ", sspfixed1[0],
                                             sspfixed1[1],
                                             sspfixed1[2])
    print("Period:", period)
    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 12,
                height   = 12) as fig:

        fig.suptitle(dynamics.get_title())
        ax1   = fig.add_subplot(2,2,1,projection='3d')
        xyz  = section.get_plane(orbit)
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
        ax2.scatter(recurrences.points2D[:, 0], recurrences.points2D[:, 1],
                   c      = 'xkcd:red',
                   marker = 'x',
                   s      = 25,
                   label  = 'Poincare Section')
        ax2.scatter(recurrences.Sorted[:, 0], recurrences.Sorted[:, 1],
                c      = 'xkcd:blue',
                marker = '+',
                s      = 25,
                label  = 'Sorted Poincare Section')
        Interpolated = recurrences.fPoincare(recurrences.create_sArray())
        ax2.scatter(Interpolated[:, 0], Interpolated[:, 1],
                c      = 'xkcd:green',
                marker = 'o',
                s      = 1,
                label  = 'Interpolated Poincare Section')
        ax2.scatter(psfixed[0], psfixed[1],
                   c      = 'xkcd:magenta',
                   marker = 'D',
                   s      = 25,
                   label  = 'psfixed')
        ax2.set_xlabel('$\\hat{x}\'$')
        ax2.set_ylabel('$z$')
        ax2.legend()

        ax3 = fig.add_subplot(2,2,3)
        ax3.scatter(recurrences.sn1, recurrences.sn2,
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = 'Sorted')
        ax3.scatter(recurrences.create_sArray(), splev(recurrences.create_sArray(), recurrences.tckReturn),
                   color  = 'xkcd:blue',
                   marker = 'o',
                   s      = 1,
                   label = 'Interpolated')
        ax3.plot(recurrences.create_sArray(), recurrences.create_sArray(),
                color     = 'xkcd:black',
                linestyle = 'dotted',
                label     = '$y=x$')
        ax3.legend()
        ax3.set_title('Arc Lengths')

        ax4 = fig.add_subplot(2,2,4, projection='3d')
        ax4.plot(sspfixedSolution[0,:], sspfixedSolution[1,:], sspfixedSolution[2,:],
                linewidth = 5,
                linestyle = 'dashed',
                c         = 'xkcd:blue',
                label     = f'Approx, period={dt:.6f}')
        ax4.scatter(sspfixedSolution[0,0], sspfixedSolution[1,0], sspfixedSolution[2,0],
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 25,
                   label = 'Start')
        ax4.scatter(sspfixedSolution[0,-1], sspfixedSolution[1,-1], sspfixedSolution[2,-1],
                   color  = 'xkcd:black',
                   marker = '+',
                   s      = 25,
                   label = 'End')
        ax4.plot(periodicOrbit[0,:], periodicOrbit[1,:], periodicOrbit[2,:],
                linewidth = 1,
                c         = 'xkcd:magenta',
                label     = f'Orbit: period={period:.6f}')
        ax4.scatter(periodicOrbit[0,0], periodicOrbit[1,0], periodicOrbit[2,0],
                   color  = 'xkcd:green',
                   marker = 'o',
                   s      = 25,
                   label = 'Refined')
        ax4.legend()

        fig.tight_layout()

    show()
