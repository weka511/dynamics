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

'''Periodic orbits and desymmetrization of the Lorenz flow'''

# Much of the code has been shamelessly stolen from Newton.py
# on page https://phys7224.herokuapp.com/grader/homework3

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import argmin, argsort, argwhere, array, cross, dot, linspace, real, size, zeros
from numpy.linalg           import norm
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
        '''Used to iterate through intersections between orbit and section'''
        for i in range(len(orbit)-1):
            u0 = self.U(orbit.orbit[:,i])
            u1 = self.U(orbit.orbit[:,i+1])
            if u0<0 and u1>0:
                ratio = abs(u0)/(abs(u0)+u1)
                yield self.refine_intersection(orbit, ratio*(orbit.t[i+1] - orbit.t[i]), orbit.orbit[:,i])

    def refine_intersection(self,orbit, dt, y):
        '''Refine an estimated intersection point '''
        dt_intersection = fsolve(lambda t: self.U(orbit.Flow(t,y)[1]), dt)[0]
        return orbit.Flow(dt_intersection, y)

    def project_to_section(self,points):
        '''Transform points on the section from (x,y,z) to coordinates embedded in surface'''
        Transformed = dot(self.ProjPoincare, points.transpose())
        Transformed =  Transformed.transpose()
        return  Transformed[:, 0:2]

class Recurrences:
    '''This class keeps track of the recurrences of the Poincare map'''
    def __init__(self,section):
        self.section   = section

    def build2D(self,crossings):
        points3D                    = array([point for _,point in crossings])
        self.points2D               = self.section.project_to_section(points3D)
        self.Sorted, ArcLengths, sn = self.sort_by_distance_from_centre(self.points2D)
        self.build_interpolated( ArcLengths, sn)
        sn1                 = sn[0:-1]
        sn2                 = sn[1:]
        isort               = argsort(sn1)
        self.sn1            = sn1[isort]
        self.sn2            = sn2[isort]
        self.tckReturn      = splrep(self.sn1,self.sn2)
        self.snPlus1        = splev(self.sArray, self.tckReturn)

    def sort_by_distance_from_centre(self,points2D):
        Distance   = squareform(pdist(points2D))
        Sorted     = points2D.copy()
        n          = size(self.points2D,0)
        ArcLengths = zeros(n)  # arclengths of the Poincare section points ordered by distance from centre
        sn                     = zeros(n)  # arclengths of the Poincare section points in dynamical order
        for k in range(n - 1):
            m                     = argmin(Distance[k, k + 1:]) + k + 1

            saved                 = Sorted[k + 1, :].copy()
            Sorted[k + 1, :] = Sorted[m, :]
            Sorted[m, :]     = saved

            saved               = Distance[:, k + 1].copy()
            Distance[:, k + 1]  = Distance[:, m]
            Distance[:, m]      = saved

            saved               = Distance[k + 1, :].copy()
            Distance[k + 1, :]  = Distance[m, :]
            Distance[m, :]      = saved

            ArcLengths[k + 1]                                = ArcLengths[k] + Distance[k, k + 1]
            sn[argwhere(points2D[:, 0] == Sorted[k + 1, 0])] = ArcLengths[k + 1]

        return  Sorted, ArcLengths, sn

    def build_interpolated(self, ArcLengths, sn, num  = 1000):
        self.tckPoincare,_ = splprep([self.Sorted[:, 0], self.Sorted[:, 1]],
                                                   u = ArcLengths,
                                                   s = 0)
        self.sArray       = linspace(min(ArcLengths), max(ArcLengths),
                                                    num = num)
        self.Interpolated = self.fPoincare(self.sArray)


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

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = DynamicsFactory.products[0],
                        help    = 'The Dynamics to be investigated')
    parser.add_argument('--dt',
                        type    = float,
                        default = 50.0,
                        help    = 'Time interval for integration')
    parser.add_argument('--fp',
                        type     = int,
                        default  = 0,
                        help    = 'Fixed point to start from')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Folder to store figures')
    parser.add_argument('--sspTemplate',
                        nargs   = 3,
                        type    = float,
                        default = [1,1,0],
                        help    = 'Template point for Poincare Section')
    parser.add_argument('--nTemplate',
                        nargs   = 3,
                        type    = float,
                        default = [1,-1,0],
                        help    = 'Normal for Poincare Section')
    return parser.parse_args()



def build_crossing_plot(crossings):
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
    dynamics    = DynamicsFactory.create(args)
    eqs         = Equilibrium.create(dynamics)
    fp          = eqs[args.fp]
    w,v         = list(fp.get_eigendirections())[0]
    orbit       = Orbit(dynamics,
                        dt          = args.dt,
                        origin      = fp,
                        direction   = real(v),
                        eigenvalue  = w)
    section     = Section(sspTemplate = args.sspTemplate,
                          nTemplate   = args.nTemplate)
    recurrences = Recurrences(section)
    recurrences.build2D(section.get_crossings(orbit))

    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics) as fig:
        ax   = fig.add_subplot(1,2,1,projection='3d')
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
        ax = fig.add_subplot(1,2,2)
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
        # ax.scatter(psfixed[0], psfixed[1],
                   # c      = 'xkcd:magenta',
                   # marker = 'D',
                   # s      = 25,
                   # label  = 'psfixed')
        ax.legend()
    show()
