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
from numpy             import array, cross, dot, meshgrid, real
from numpy.linalg      import norm
from utils             import Figure, Timer

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

    def get_plane(self,
                  xmin = -10,
                  xmax = 10,
                  ymin = -10,
                  ymax = 10,
                  zmin = -10,
                  zmax = 10):
        d = -self.sspTemplate.dot(self.nTemplate) # snarfed from https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
        if self.nTemplate[0]!= 0:
            yy, zz = meshgrid(range(int(ymin),int(ymax)), range(int(zmin),int(zmax)))
            xx     = (-self.nTemplate[1] * yy - self.nTemplate[2] * zz - d) /self.nTemplate[0]
        elif self.nTemplate[1]!= 0:
            zz, xx = meshgrid(range(int(zmin),int(zmax)), range(int(xmin),int(xmax)))
            yy     = (-self.nTemplate[2] * zz - self.nTemplate[0] * xx - d) /self.nTemplate[1]
        elif self.nTemplate[2]!= 0:
            xx, yy = meshgrid(range(int(xmin),int(xmax)), range(int(ymin),int(ymax)))
            zz     = (-self.nTemplate[0] * xx - self.nTemplate[1] * yy - d) /self.nTemplate[2]

        return xx,yy,zz

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
    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics) as fig:
        ax        = fig.add_subplot(1,1,1,projection='3d')
        xx,yy,zz  = section.get_plane(xmin = min(orbit.orbit[0,:]),
                                      xmax = max(orbit.orbit[0,:]),
                                      ymin = min(orbit.orbit[1,:]),
                                      ymax = max(orbit.orbit[1,:]),
                                      zmin = min(orbit.orbit[2,:]),
                                      zmax = max(orbit.orbit[2,:]))
        ax.plot_surface(xx,yy,zz,
                        color = 'xkcd:blue',
                        alpha = 0.5)
        ax.plot(orbit.orbit[0,:],orbit.orbit[1,:],orbit.orbit[2,:],
                color = 'xkcd:green')

    show()
