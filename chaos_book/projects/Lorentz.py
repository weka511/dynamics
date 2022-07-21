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
from matplotlib.pyplot      import figure, rcParams, savefig, show, suptitle
from numpy                  import arange, argmin, argsort, argwhere, array, cos, cross, dot, linspace, pi, sin, size, sqrt, zeros
from numpy.random           import rand
from os.path                import join, basename, split
from pathlib                import Path
from scipy.integrate        import solve_ivp
from scipy.interpolate      import splev, splprep, splrep
from scipy.linalg           import eig, norm
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform

class Dynamics:

    def get_title(self):
        return fr'{self.name} $\sigma=${self.sigma}, $\rho=${self.rho}, b={self.b}'

    def get_x_label(self):
        return 'x'

    def get_y_label(self):
        return 'y'

    def get_z_label(self):
        return 'z'


class Lorentz(Dynamics):
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        self.sigma = sigma
        self.rho   = rho
        self.b     = b
        self.name  = 'Lorentz'

    def create_eqs(self):
        eq0 = [0,0,0]
        if self.rho<1:
            return array([eq0])
        else:
            x = sqrt(self.b*(self.rho-1))
            return array([eq0,
                         [x,x,self.rho-1],
                         [-x,-x,self.rho-1]])

    def velocity(self, t,stateVec):
        '''
        return the velocity field of Lorentz system.
        stateVec : the state vector in the full space. [x, y, z]
        t : time is used since solve_ivp() requires it.
        '''

        x = stateVec[0]
        y = stateVec[1]
        z = stateVec[2]

        return array([self.sigma * (y-x),
                      self.rho*x - y - x*z,
                      x*y - self.b*z])

class PseudoLorentz(Dynamics):
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        self.sigma = sigma
        self.rho   = rho
        self.b     = b
        self.name     = 'Pseudo Lorentz'

    def create_eqs(self):
        eq0 = [0,0,0]
        eq1 = [0,2*self.b*(self.rho-1),self.rho-1]
        return  array([eq0,eq1])

    def velocity(self,t,stateVec):
        u = stateVec[0]
        v = stateVec[1]
        z = stateVec[2]
        N = sqrt(u**2 + v**2)
        return array([-(self.sigma+1)*u + (self.sigma-self.rho)*v + (1-self.sigma)*N + v*z,
                      (self.rho-self.sigma)*u - (self.sigma+1)*v + (self.rho+self.sigma)*N - u*z -u*N,
                      v/2 - self.b*z])

    def get_x_label(self):
        return 'u'

    def get_y_label(self):
        return 'v'

class Integrator:
    def __init__(self,dynamics):
        self.dynamics = dynamics


    def integrate(self,init_x, dt, nstp=1):
        '''
        The integrator of the Lorentz system.
        init_x: the intial condition
        dt : time step
        nstp: number of integration steps.

        return : a [ nstp x 3 ] vector
        '''

        bunch = solve_ivp(dynamics.velocity, (0, dt), init_x, t_eval=arange(0,dt,dt/nstp))
        if bunch.status==0:
            return bunch.t, bunch.y
        else:
            raise(Exception(f'{bunch.status} {bunch.message}'))



    def Flow(self,deltat,y):
        bunch = solve_ivp(dynamics.velocity, (0, deltat), y)
        return bunch.t[1],bunch.y[:,1]

class PoincareSection:
    ''' This class represents a Poincare Section'''
    @staticmethod
    def zRotation(theta):
        '''
        Rotation matrix about z-axis
        Input:
        theta: Rotation angle (radians)
        Output:
        Rz: Rotation matrix about z-axis
        '''
        return array([[cos(theta), -sin(theta), 0],
                      [sin( theta), cos(theta),  0],
                      [0,          0,           1]],
                     float)

    def __init__(self,dynamics,integrator,
                 sspTemplate = None,
                 nTemplate   = None,
                 theta       = 0.0,
                 e_x         = array([1, 0, 0], float)):
        self.sspTemplate = dot(PoincareSection.zRotation(theta), e_x)  if len(sspTemplate)==1 else sspTemplate
        self.nTemplate   = dot(PoincareSection.zRotation(pi/2), self.sspTemplate) if len(nTemplate)==1 else nTemplate
        self.e_1         =array([0,1,0])                         # FIXME
        self.e_2         = cross(self.nTemplate, self.e_1)
        self.integrator  = integrator
        self.dynamics    = dynamics

    def U(self, ssp):
        '''
        Plane equation for the Poincare section hyperplane which includes z-axis
        and makes an angle theta with the x-axis see ChaosBook ver. 14, fig. 3.2
        Inputs:
          ssp: State space point at which the Poincare hyperplane equation will be
               evaluated
        Outputs:
          U: Hyperplane equation which should be satisfied on the Poincare section
           U = (ssp - sspTemplate) . nTemplate (see ChaosBook ver. 14, eq. 3.6)
        '''
        return dot((ssp - self.sspTemplate),self.nTemplate)

    def Flow(self,y0,dt):
        _,y = self.integrator.integrate(y0,dt)
        return y[0]

    def interpolate(self,dt0, y0):
        return self.integrator.Flow(fsolve(lambda t: self.U(self.Flow(y0, t)), dt0)[0],
                                    y0)


    def interections(self,ts, orbit):
        _,n = orbit.shape
        for i in range(n-1):
            if self.U(orbit[:,i])<0 and self.U(orbit[:,i+1])>0:
                yield self.interpolate(0.5*(ts[i+1]-ts[i]), orbit[:,i])

    def project(self,point):
        projection_normal = dot(self.nTemplate,point)
        projected         = point -  projection_normal*self.nTemplate
        x                 = dot(projected,self.e_1)
        y                 = dot(projected,self.e_2)
        return x,y

    def create_arclengths(self):
        ps                    = array([self.project(point) for _,point in self.interections(ts,orbit)])
        Distance              = squareform(pdist(ps))
        SortedPoincareSection = ps.copy()
        ArcLengths            = zeros(size(SortedPoincareSection, 0))
        sn                    = zeros(size(ps, 0)) # the arclengths of the Poincare section points keeping their dynamical order for use in the return map
        for k in range(size(SortedPoincareSection, 0) - 1):
            m             = argmin(Distance[k, k + 1:]) + k + 1     #Find the element which is closest to the kth point:
            dummyPoincare = SortedPoincareSection[k + 1, :].copy()  #Hold the (k+1)th row in the dummy vector:
            SortedPoincareSection[k + 1, :] = SortedPoincareSection[m, :]  #Replace (k+1)th row with the closest point:
            SortedPoincareSection[m, :] = dummyPoincare    #Assign the previous (k+1)th row to the mth row:

            #Rearrange the distance matrix according to the new form of the
            #SortedPoincareSection array:
            dummyColumn = Distance[:, k + 1].copy()  # Hold (k+1)th column of the
                                                     # distance matrix in a dummy
                                                     # array
            Distance[:, k + 1] = Distance[:, m]  # Assign mth column to kth
            Distance[:, m] = dummyColumn

            dummyRow = Distance[k + 1, :].copy()  # Hold (k+1)th row in a dummy array
            Distance[k + 1, :] = Distance[m, :]
            Distance[m, :] = dummyRow

            ArcLengths[k + 1] = ArcLengths[k] + Distance[k, k + 1] #Assign the arclength of (k+1)th element:
            #Find this point in the PoincareSection array and assign sn to its
            #corresponding arclength:
            sn[argwhere(ps[:, 0] == SortedPoincareSection[k + 1, 0])] = ArcLengths[k + 1]

        #Parametric spline interpolation to the Poincare section:
        self.tckPoincare, u = splprep([SortedPoincareSection[:, 0], SortedPoincareSection[:, 1]],
                                      u = ArcLengths,
                                      s = 0)
        sArray = linspace(min(ArcLengths), max(ArcLengths), 1000)
        #Evaluate the interpolation:
        InterpolatedPoincareSection = self.fPoincare(sArray)

        sn1       = sn[0:-1]
        sn2       = sn[1:]
        isort     = argsort(sn1)
        sn1       = sn1[isort]  # sort radii1
        sn2       = sn2[isort]  # sort radii2
        tckReturn = splrep(sn1,sn2)
        snPlus1   = splev(sArray, tckReturn)
        return sn1,sn2,sArray,snPlus1

    def fPoincare(self,s):
        '''
        Parametric interpolation to the Poincare section
        Inputs:
        s: Arc length which parametrizes the curve, a float or dx1-dim numpy
           array
        Outputs:
        xy = x and y coordinates on the Poincare section, 2-dim numpy array
           or (dx2)-dim numpy array
        '''
        interpolation = splev(s, self.tckPoincare)
        return array([interpolation[0], interpolation[1]], float).transpose()

def plot_poincare(ax,section,ts,orbit,s=1):
    for t,point in section.interections(ts,orbit):
        ax.scatter(point[0],point[1],point[2],
                   c      = 'xkcd:green',
                   s      = s,
                   marker = 'o')

    ax.scatter(point[0],point[1],point[2],
               c      = 'xkcd:green',
               s      = s,
               label  = r'Poincar\'e return',
               marker = 'o')

def parse_args():
    parser  = ArgumentParser(description = __doc__)
    parser.add_argument('action', type = int)
    parser.add_argument('--fp',   type = int, default = 1)
    parser.add_argument('--figs', default = './figs')
    return parser.parse_args()

if __name__ == '__main__':
    rcParams['text.usetex'] = True
    args                    = parse_args()
    fig                     = figure(figsize=(12,12))

    if args.action==1:
        dynamics      = Lorentz()
        integrator    = Integrator(dynamics)
        EQs           = dynamics.create_eqs()
        section       = PoincareSection(dynamics,integrator,
                                        sspTemplate = EQs[args.fp],
                                        nTemplate   = array([1,0,0]))

        x0            = EQs[0,:] + 0.001*rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        ts,orbit      = integrator.integrate(x0, 50.0, nstp)

        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                c          = 'xkcd:blue',
                label      = 'Orbit',
                markersize = 1)
        ax.scatter(EQs[0,0], EQs[0,1], EQs[0,2], marker='o', c='xkcd:red', label='EQ0')
        ax.scatter(EQs[1,0], EQs[1,1], EQs[1,2], marker='1', c='xkcd:red', label='EQ1')
        ax.scatter(EQs[2,0], EQs[2,1], EQs[2,2], marker='2', c='xkcd:red', label='EQ2')
        plot_poincare(ax,section,ts,orbit)
        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())


    if args.action==2:
        dynamics   = PseudoLorentz()
        integrator = Integrator(dynamics)
        EQs        = dynamics.create_eqs()
        section    = PoincareSection(dynamics,integrator,
                                        sspTemplate = EQs[args.fp],
                                        nTemplate   = array([1,0,0]))
        x0         = EQs[0,:] + 0.001*rand(3)
        dt         = 0.001
        nstp       = 50.0/dt
        ts,orbit   = integrator.integrate(x0, 50.0, nstp)

        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                markersize = 1,
                c          = 'xkcd:blue',
                label      = 'Orbit')
        ax.scatter(EQs[0,0], EQs[0,1], EQs[0,2], marker='o', c='xkcd:red', label='EQ0')
        ax.scatter(EQs[1,0], EQs[1,1], EQs[1,2], marker='1', c='xkcd:red', label='EQ1')
        plot_poincare(ax,section,ts,orbit)
        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())


    if args.action==3:
        dynamics      = Lorentz()
        integrator    = Integrator(dynamics)
        EQs           = dynamics.create_eqs()
        section       = PoincareSection(dynamics,integrator,
                                        sspTemplate = EQs[args.fp],
                                        nTemplate   = array([1,0,0]))

        x0            = EQs[0,:] + 0.001*rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        ts,orbit      = integrator.integrate(x0, 50.0, nstp)
        sn1,sn2,sArray,snPlus1 = section.create_arclengths()

        ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        ax.scatter(sn1, sn2,
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = 'Sorted')
        ax.scatter(sArray, snPlus1,
                   color  = 'xkcd:blue',
                   marker = 'o',
                   s      = 1,
                   label = 'Interpolated')
        ax.plot(sArray, sArray,
                color = 'xkcd:black',
                linestyle = 'dotted',
                label     = '$y=x$')
    ax.legend()
    savefig(join(args.figs,f'{Path(__file__).stem}{args.action}'))
    show()
