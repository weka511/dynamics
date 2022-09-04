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

This file contains classes that model a Poincare Section and
assist in finding recurrences and cycles. It includes test code
for the Rossler equation.
'''

# Much of the code has been shamelessly stolen from Newton.py
# on page https://phys7224.herokuapp.com/grader/homework3

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import append, argmin, argsort, argwhere, array, cross, dot, linspace, real, size, zeros
from numpy.linalg           import inv, norm
from scipy.interpolate      import splev, splprep, splrep
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform
from utils                  import get_plane, Figure

class Section:
    ''' This class represents a Poincare Section'''
    def __init__(self,
                 sspTemplate = array([1,1,0]),
                 nTemplate   = array([1,-1,0])):
        self.sspTemplate  = sspTemplate/norm(sspTemplate)
        self.nTemplate    = nTemplate/norm(nTemplate)
        self.ProjPoincare = array([self.sspTemplate,
                                   cross(self.sspTemplate,self.nTemplate),
                                   self.nTemplate],
                                  float)

    def U(self, ssp):
        '''
        Plane equation for the Poincare section: see ChaosBook ver. 14, fig. 3.2.

        Inputs:
          ssp: State space point at which the Poincare hyperplane equation will be
               evaluated
        Outputs:
          U: Hyperplane equation which should be satisfied on the Poincare section
           U = (ssp - sspTemplate) . nTemplate (see ChaosBook ver. 14, eq. 3.6)
        '''
        return dot((ssp - self.sspTemplate),self.nTemplate)

    def get_plane(self,orbit,num=50):
        '''Used to plot section as a surface'''
        return get_plane(sspTemplate = self.sspTemplate,
                         nTemplate   = self.nTemplate,
                         limits      = [linspace(m, M, num = num) for m,M in zip(orbit.orbit.min(axis=1),orbit.orbit.max(axis=1))])


    def get_crossings(self,orbit):
        t0 = 0
        t_events = orbit.t_events[0]
        y_events = orbit.y_events[0]
        for i in range(len(t_events)):
            yield t_events[i]-t0,array(y_events[i])
            t0 = t_events[i]

    def project_to_section(self,points):
        '''Transform points on the section from (x,y,z) to coordinates embedded in surface'''
        return  dot(self.ProjPoincare, points.transpose()).transpose()[:, 0:2]

    def project_to_space(self,point):
        '''Transform a point embedded in surface back to (x,y,z) coordinates '''
        return dot(append(point, 0.0), self.ProjPoincare)

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
    return parser.parse_args()

def build_crossing_plot(crossings):
    '''Used to construct plot for crossing section'''
    xs = []
    ys = []
    zs = []
    for _,ssp in crossings:
        xs.append(ssp[0])
        ys.append(ssp[1])
        zs.append(ssp[2])
    return xs,ys,zs



if __name__=='__main__':
    args        = parse_args()
    section     = Section(sspTemplate = args.sspTemplate,
                          nTemplate   = args.nTemplate)
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]
    event       = lambda t,y: section.U(y)
    event.direction = 1.0
    orbit       = Orbit(dynamics,
                        dt          = args.dt,
                        origin      = fp,
                        direction   = real(v),
                        eigenvalue  = w,
                        events      = event)

    recurrences = Recurrences(section)
    recurrences.build2D(section.get_crossings(orbit))
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
        ax   = fig.add_subplot(2,2,1,projection='3d')
        xyz  = section.get_plane(orbit)
        ax.plot_surface(xyz[0,:], xyz[1,:], xyz[2,:],
                        color = 'xkcd:blue',
                        alpha = 0.5)
        ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                color = 'xkcd:green',
                label = f'{dynamics.name}')
        xs,ys,zs  = build_crossing_plot(section.get_crossings(orbit))
        ax.scatter(xs,ys,zs,
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
        ax.scatter(psfixed[0], psfixed[1],
                   c      = 'xkcd:magenta',
                   marker = 'D',
                   s      = 25,
                   label  = 'psfixed')
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

        ax = fig.add_subplot(2,2,4, projection='3d')
        ax.plot(sspfixedSolution[0,:], sspfixedSolution[1,:], sspfixedSolution[2,:],
                linewidth = 5,
                linestyle = 'dashed',
                c         = 'xkcd:blue',
                label     = f'Approx, period={dt:.6f}')
        ax.scatter(sspfixedSolution[0,0], sspfixedSolution[1,0], sspfixedSolution[2,0],
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 25,
                   label = 'Start')
        ax.scatter(sspfixedSolution[0,-1], sspfixedSolution[1,-1], sspfixedSolution[2,-1],
                   color  = 'xkcd:black',
                   marker = '+',
                   s      = 25,
                   label = 'End')
        ax.plot(periodicOrbit[0,:], periodicOrbit[1,:], periodicOrbit[2,:],
                linewidth = 1,
                c         = 'xkcd:magenta',
                label     = f'Orbit: period={period:.6f}')
        ax.scatter(periodicOrbit[0,0], periodicOrbit[1,0], periodicOrbit[2,0],
                   color  = 'xkcd:green',
                   marker = 'o',
                   s      = 25,
                   label = 'Refined')
        ax.legend()

        fig.tight_layout()

    show()
