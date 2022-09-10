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

This file contains classes that model a Poincare Section.
'''

from argparse               import ArgumentParser
from dynamics               import DynamicsFactory, Equilibrium, Orbit
from matplotlib.pyplot      import show
from numpy                  import append, array, cross, dot, linspace, real, searchsorted
from scipy.linalg           import norm
from sys                    import float_info
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

    def get_plane(self,orbit,
                  t0  = 0,
                  t1  = float_info.max,
                  num = 50):
        '''
        Used to plot section as a surface

        Parameters:
            orbit    An orbit: the surface will be of compable size to orbit
            t0       If specified, the size will depend only on points subsequent to this time
            t1       If specified, the size will depend only on points prior to this time
            num      Controls granularity of area to be plotted
        '''
        i0        = searchsorted(orbit.t,t0)
        i1        = searchsorted(orbit.t,t1)
        yy_reduced = orbit.y[:,i0:i1]
        return get_plane(sspTemplate = self.sspTemplate,
                         nTemplate   = self.nTemplate,
                         limits      = [linspace(m, M, num = num) for m,M in zip(yy_reduced.min(axis=1),yy_reduced.max(axis=1))])



    def project_to_section(self,points):
        '''Transform points on the section from (x,y,z) to coordinates embedded in surface'''
        return  dot(self.ProjPoincare, points.transpose()).transpose()[:, 0:2]

    def project_to_space(self,point):
        '''Transform a point embedded in surface back to (x,y,z) coordinates '''
        return dot(append(point, 0.0), self.ProjPoincare)

    def establish_crossings(self,
                            direction = 1.0,
                            terminal = False):
        '''
        Establish the definition of an orbit crossing section.

        Used by Orbit to define events for solve_ivp.
        '''
        event           = lambda t,y: self.U(y)
        event.direction = direction
        event.terminal  = terminal
        return event

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

    with Figure(figs     = args.figs,
                file     = __file__,
                dynamics = dynamics,
                width    = 12,
                height   = 12) as fig:

        fig.suptitle(dynamics.get_title())
        ax   = fig.add_subplot(1,1,1,projection='3d')
        xyz  = section.get_plane(orbit)
        crossings = array([ssp for _,ssp in orbit.get_events()])
        ax.plot_surface(xyz[0,:], xyz[1,:], xyz[2,:],
                        color = 'xkcd:blue',
                        alpha = 0.5)
        ax.plot(orbit.y[0,:],orbit.y[1,:],orbit.y[2,:],
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

    show()
