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

from argparse          import ArgumentParser
from dynamics          import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot import show
from numpy             import array, cross, dot, linspace, meshgrid, real
from numpy.linalg      import norm
from scipy.optimize    import fsolve
from utils             import get_plane, Figure, Timer

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
    def get_plane(self,orbit):
        '''Used to plot section as a surface'''
        m0 = orbit.orbit[:,:].min()
        m1 = orbit.orbit[:,:].max()
        return get_plane(sspTemplate = self.sspTemplate,
                         nTemplate   = self.nTemplate,
                         xs          = linspace(m0,m1,50),
                         ys          = linspace(m0,m1,50))


    def crossings(self,orbit):
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

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = DynamicsFactory.products[0])
    parser.add_argument('--dt',
                        type    = float,
                        default = 50.0)
    parser.add_argument('--eq',
                        type     = int,
                        default  = 0)
    parser.add_argument('--figs',
                        default = './figs')
    return parser.parse_args()

if __name__=='__main__':
    args     = parse_args()
    dynamics = DynamicsFactory.create(args)
    eqs      = Equilibrium.create(dynamics)
    eq       = eqs[args.eq]
    w,v      = list(eq.get_eigendirections())[0]
    orbit    = Orbit(dynamics,
                    dt          = args.dt,
                    origin      = eqs[args.eq],
                    direction   = real(v),
                    eigenvalue  = w)
    section  = Section()
    xs = []
    ys = []
    zs = []
    for _,ssp in section.crossings(orbit):
        xs.append(ssp[0])
        ys.append(ssp[1])
        zs.append(ssp[2])

    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics) as fig:
        ax        = fig.add_subplot(1,1,1,projection='3d')
        xx,yy,zz  = section.get_plane(orbit)
        ax.plot_surface(xx,yy,zz,
                        color = 'xkcd:blue',
                        alpha = 0.5)
        ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                color = 'xkcd:green',
                label = f'{dynamics.name}')
        ax.scatter(xs,ys,zs,
                   color = 'xkcd:red',
                   s     = 1,
                   label = 'Crossings')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    show()
