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

from abc                    import ABC,abstractmethod
from argparse               import ArgumentParser
from matplotlib.markers     import MarkerStyle
from matplotlib.pyplot      import figlegend, figure, rcParams, savefig, show, suptitle, tight_layout
from numpy                  import append, arange, argmin, argsort, argwhere, array, cos, cross, dot, identity, \
                                   iinfo, int64, linspace, pi, real, reshape, sin, size, sqrt, zeros
from numpy.linalg           import eig, inv, norm
from os.path                import join, basename, split
from pathlib                import Path
from scipy.integrate        import solve_ivp
from scipy.interpolate      import splev, splprep, splrep
from scipy.linalg           import eig, norm
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform
from sys                    import exc_info

class Dynamics(ABC):
    '''This abstract class represents the dynamics, i.e. the differential equation.'''
    def __init__(self, name = None,
                 d          = 3):
        self.name = name
        self.d    = d

    @abstractmethod
    def find_equilibria(self):
        ...

    @abstractmethod
    def Velocity(self, t,stateVec):
        ...

    @abstractmethod
    def StabilityMatrix(self,stateVec):
        ...

    def JacobianVelocity(self,t, sspJacobian):
        '''
        Velocity function for the Jacobian integration

        Inputs:
            sspJacobian: (d+d^2)x1 dimensional state space vector including both the
                         state space itself and the tangent space
            t: Time. Has no effect on the function, we have it as an input so that our
               ODE would be compatible for use with generic integrators from
               scipy.integrate

        Outputs:
            velJ = (d+d^2)x1 dimensional velocity vector
        '''

        ssp            = sspJacobian[0:self.d]
        J              = sspJacobian[self.d:].reshape((self.d, self.d))
        velJ           = zeros(size(sspJacobian))
        velJ[0:self.d] = self.Velocity(t, ssp)
        velTangent     = dot(self.StabilityMatrix(ssp), J)
        velJ[self.d:]  = reshape(velTangent, self.d**2)
        return velJ

    def get_title(self):
        '''For display on plots'''
        return fr'{self.name} $\sigma=${self.sigma}, $\rho=${self.rho}, b={self.b}'

    def get_x_label(self):
        '''For display on plots'''
        return 'x'

    def get_y_label(self):
        '''For display on plots'''
        return 'y'

    def get_z_label(self):
        '''For display on plots'''
        return 'z'

    def get_start_on_unstable_manifold(self,eq0,
                                       eps = 1e-6):
        '''Initial condition as a slight perturbation to specified fixed point in the direction of eigenvector with largest eigenvalue'''
        Aeq0                      = self.StabilityMatrix(eq0)
        eigenValues, eigenVectors = eig(Aeq0)
        # if eigenValues[0]<0:
            # raise Exception(f'EQ {eq0} is stable {eigenValues}, so there is no unstable manifold')
        v1 = real(eigenVectors[:, 0])
        return eq0 + eps * v1  / norm(v1)

class Lorentz(Dynamics):
    '''Dynamics of Lorentz Equation'''
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        super().__init__('Lorentz')
        self.sigma = sigma
        self.rho   = rho
        self.b     = b

    def find_equilibria(self):
        if self.rho<1:
            return array([0,0,0])
        else:
            x = sqrt(self.b*(self.rho-1))
            return array([[0,  0,  0],
                          [x,  x,  self.rho-1],
                          [-x, -x, self.rho-1]])

    def Velocity(self, t,stateVec):
        '''
        Return the Velocity field of Lorentz system.
        stateVec : the state vector in the full space. [x, y, z]
        t : time is used since solve_ivp() requires it.
        '''

        x,y,z = stateVec

        return array([self.sigma * (y-x),
                      self.rho*x - y - x*z,
                      x*y - self.b*z])

    def StabilityMatrix(self,stateVec):
        '''
        return the stability matrix at a state point.
        stateVec: the state vector in the full space. [x, y, z]
        '''

        x,y,z = stateVec

        return array([
            [-self.sigma, self.sigma, 0],
            [self.rho-z,  -1 ,       -x],
            [y,           x,         -self.b]
        ])



class ProtoLorentz(Dynamics):
    '''Dynamics of Proto-Lorentz Equation'''
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        super().__init__('Proto-Lorentz')
        self.sigma = sigma
        self.rho   = rho
        self.b     = b

    def find_equilibria(self):
        eq0 = [0,0,0]
        eq1 = [0,2*self.b*(self.rho-1),self.rho-1]
        return  array([eq0,eq1])

    def Velocity(self,t,stateVec):
        u,v,z = stateVec
        N     = sqrt(u**2 + v**2)
        return array([-(self.sigma+1)*u + (self.sigma-self.rho)*v + (1-self.sigma)*N + v*z,
                      (self.rho-self.sigma)*u - (self.sigma+1)*v + (self.rho+self.sigma)*N - u*z -u*N,
                      v/2 - self.b*z])

    def StabilityMatrix(self,stateVec):
        '''
        return the stability matrix at a state point.
        stateVec: the state vector in the full space. [x, y, z]
        '''

        u,v,z = stateVec
        N = sqrt(u**2 + v**2)
        if N>0:
            return array([
                [-(self.sigma+1) + (1-self.sigma)*u/N,
                 self.sigma -self.rho + (1-self.sigma)*v/N,
                 v],
                [self.rho-self.sigma + (self.rho+self.sigma)*u/N - z -N - u**2/N,
                 -(self.sigma+1) +  (self.rho+self.sigma)*v/N - u*v/N,
                 -u],
                [0,           0.5,         -self.b]
            ])
        else:
            return ([[-2*self.sigma, 1-self.rho,   0],
                     [2*self.rho -z, self.rho-1,   0],
                     [0,             0.5,         -self.b]])

    def get_x_label(self):
        return 'u'

    def get_y_label(self):
        return 'v'

class Rossler(Dynamics):
    def __init__(self,
                 a = 0.2,
                 b = 0.2,
                 c = 5.7):
        super().__init__('Rossler')
        self.a = a
        self.b = b
        self.c = c

    def find_equilibria(self):
        term1 = 0.5*self.c/self.a
        term2 = sqrt(term1**2-self.b/self.a)
        y1    = term1 + term2
        y2    = term1 - term2
        return  array([[-self.a*y1,y1,-y1],[-self.a*y2,y2,-y2]])

    def Velocity(self, t,stateVec):
        '''
        Return the Velocity field of Lorentz system.
        stateVec : the state vector in the full space. [x, y, z]
        t : time is used since solve_ivp() requires it.
        '''

        x,y,z = stateVec

        dxdt = - y - z
        dydt = x + self.a * y
        dzdt = self.b + z * (x - self.c)

        return array([dxdt, dydt, dzdt], float)  # Velocity vector



    def StabilityMatrix(self,ssp):
        '''
        Stability matrix for the Rossler flow

        Inputs:
            ssp: State space vector. dx1 NumPy array: ssp = [x, y, z]
        Outputs:
            A: Stability matrix evaluated at ssp. dxd NumPy array
               A[i, j] = del Velocity[i] / del ssp[j]
        '''

        x, y, z = ssp

        return array([[0, -1, -   1],
                      [1, self.a, 0],
                      [z, 0,      x - self.c]],
                     float)

    def get_title(self):
        return 'Rossler'

class DynamicsFactory:
    '''Factory class for instantiating Dynamics '''
    products = ['Lorentz', 'ProtoLorentz', 'Rossler']

    @classmethod
    def create(cls,args):
        if args.dynamics == 'Lorentz':
            return Lorentz()
        if args.dynamics == 'ProtoLorentz':
            return ProtoLorentz()
        if args.dynamics == 'Rossler':
            return Rossler()
        raise Exception(f'Unknown dynamics: {args.dynamics}')

class Integrator:
    '''This class is used to integrate  the ODEs for a specified Dynamics'''
    def __init__(self,dynamics,
                method = 'RK45'): # I was using Radau, which seemed OK for Rossler but not Lorentz
        self.dynamics = dynamics
        self.method   = method

    def integrate(self,init_x, dt, nstp=1):
        '''
        The integrator of the Lorentz system.
        init_x: the intial condition
        dt : time step
        nstp: number of integration steps.

        return : a [ nstp x 3 ] vector
        '''

        solution = solve_ivp(dynamics.Velocity, (0, dt), init_x,
                          t_eval = arange(0,dt,dt/nstp),
                          method = self.method) # Need this to close cycles
        if solution.status==0:
            return solution.t, solution.y
        else:
            raise(Exception(f'Integration error: {solution.status} {solution.message}'))



    def Flow(self,deltat,y):
        '''Used to integrate for a single step'''
        solution = solve_ivp(dynamics.Velocity, (0, deltat), y,method=self.method)
        return solution.t[1],solution.y[:,1]

    def Jacobian(self,ssp, t):
        '''
        Jacobian function for the trajectory started on ssp, evolved for time t

        Inputs:
            ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
            t: Integration time
        Outputs:
            J: Jacobian of trajectory f^t(ssp). dxd NumPy array
        '''

        Jacobian0                  = identity(dynamics.d)
        sspJacobian0               = zeros(dynamics.d + dynamics.d ** 2)
        sspJacobian0[0:dynamics.d] = ssp
        sspJacobian0[dynamics.d:]  = reshape(Jacobian0, dynamics.d**2)
        solution                   = solve_ivp(self.dynamics.JacobianVelocity, (0, t), sspJacobian0,method=self.method)

        if solution.status==0:
            assert  t == solution.t[-1]
            sspJacobianSolution = solution.y[:,-1]
            return sspJacobianSolution[3:].reshape((3, 3))
        else:
            raise(Exception(f'Integration error: {solution.status} {solution.message}'))



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
                 sspTemplate = array([1,1,0]),
                 nTemplate   = array([1,-1,0])):
        self.sspTemplate  = sspTemplate/norm(sspTemplate)
        self.nTemplate    = nTemplate/norm(nTemplate)
        third_axis        = cross(self.sspTemplate,self.nTemplate)
        self.ProjPoincare = array([self.sspTemplate,
                                   third_axis/norm(third_axis),
                                   self.nTemplate],
                                  float)
        self.integrator  = integrator
        self.dynamics    = dynamics

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

    def project_to_section(self,points):
        '''Transform points on the section from (x,y,z) to coordinates embedded in surface'''
        Transformed = dot(self.ProjPoincare, points.transpose())
        Transformed =  Transformed.transpose()
        return  Transformed[:, 0:2]

    def project_to_space(self,point):
        '''Transform a point embedded in surface back to (x,y,z) coordinates '''
        return dot(append(point, 0.0), self.ProjPoincare)

    def Flow(self,y0,dt):
        _,y = self.integrator.integrate(y0,dt)
        _,n = y.shape
        assert n==1
        return y[0]

    def interpolate(self,dt0, y0):
        return self.integrator.Flow(fsolve(lambda t: self.U(self.Flow(y0, t)), dt0)[0], y0)

    def intersections(self,ts, orbit):
        '''Used to iterate through intersections between orbit and section'''
        _,n = orbit.shape
        for i in range(n-1):
            if self.U(orbit[:,i])<0 and self.U(orbit[:,i+1])>0:
                yield self.interpolate(0.5*(ts[i+1]-ts[i]), orbit[:,i])




class Recurrences:
    '''This class keeps track of the recurrences of the Poincare nap'''
    def __init__(self,section,ts,orbit,
                 filter = lambda point: True):
        self.section               = section
        intersections              = [point for _,point in section.intersections(ts,orbit) if filter(point)]
        if len(intersections)==0: return
        self.Section               = section.project_to_section(array(intersections))
        Distance                   = squareform(pdist(self.Section))
        self.SortedPoincareSection = self.Section.copy()
        ArcLengths                 = zeros(size(self.SortedPoincareSection, 0))
        sn                         = zeros(size(self.Section, 0))
        for k in range(size(self.SortedPoincareSection, 0) - 1):
            index_closest_point_to_k        = argmin(Distance[k, k + 1:]) + k + 1

            saved_poincare_row                                      = self.SortedPoincareSection[k + 1, :].copy()
            self.SortedPoincareSection[k + 1, :]                    = self.SortedPoincareSection[index_closest_point_to_k, :]
            self.SortedPoincareSection[index_closest_point_to_k, :] = saved_poincare_row

            #Rearrange the distance matrix according to the new form of the SortedPoincareSection array:
            saved_column                          = Distance[:, k + 1].copy()
            Distance[:, k + 1]                    = Distance[:, index_closest_point_to_k]
            Distance[:, index_closest_point_to_k] = saved_column

            saved_row                              = Distance[k + 1, :].copy()
            Distance[k + 1, :]                     = Distance[index_closest_point_to_k, :]
            Distance[index_closest_point_to_k, :]  = saved_row

            ArcLengths[k + 1]                      = ArcLengths[k] + Distance[k, k + 1]
            index_in_arc_lengths                   = argwhere(self.Section[:, 0] == self.SortedPoincareSection[k + 1, 0])
            sn[index_in_arc_lengths]               = ArcLengths[k + 1]

        #Parametric spline interpolation to the Poincare section:
        self.tckPoincare, u              = splprep([self.SortedPoincareSection[:, 0], self.SortedPoincareSection[:, 1]],
                                                   u = ArcLengths,
                                                   s = 0)
        self.sArray                      = linspace(min(ArcLengths), max(ArcLengths), 1000)
        self.InterpolatedPoincareSection = self.fPoincare(self.sArray)

        sn1            = sn[0:-1]
        sn2            = sn[1:]
        isort          = argsort(sn1)
        self.sn1       = sn1[isort]
        self.sn2       = sn2[isort]
        self.tckReturn = splrep(self.sn1,self.sn2)
        self.snPlus1   = splev(self.sArray, self.tckReturn)

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

    def get_fixed(self, s0=10.0):
        sfixed = fsolve(lambda r: splev(r, self.tckReturn) - r, s0)[0]
        return  sfixed,self.section.project_to_space(self.fPoincare(sfixed))

    def solve(self, Tnext,  sspfixed,
              integrator = None,
              dynamics   = None,
              tol        = 1e-9,
              kmax       = 20):
        period              = Tnext
        error               = zeros(dynamics.d+1)
        error[0:dynamics.d] = integrator.integrate(sspfixed,period)[0] - sspfixed
        Newton              = zeros((dynamics.d+1, dynamics.d+1))
        print(f'Iteration {0} {error}')
        for k in range(kmax):
            Newton[0:dynamics.d, 0:dynamics.d] = identity(dynamics.d) - integrator.Jacobian(sspfixed,period)
            Newton[0:dynamics.d, dynamics.d]   = - dynamics.Velocity(period,sspfixed)
            Newton[dynamics.d, 0:dynamics.d]   = self.section.nTemplate
            Delta                              = dot(inv(Newton), error)
            sspfixed                           = sspfixed + Delta[0:dynamics.d]
            period                             = period + Delta[dynamics.d]
            error[0:dynamics.d]                = integrator.Flow(period,sspfixed)[0] - sspfixed

            print(f'Iteration {k} {error}')
            if max(abs(error)) > tol: return period, sspfixed
        Exception(f'Failed to converge within {tol} after {kmax} iterations: final error={max(abs(error))}')

    def getTnext(self,sspfixed):
        return fsolve(lambda dt: self.section.U(integrator.integrate(sspfixed,dt)[0]),
                              args.tFinal / size(self.Section, 0),xtol=1e-9)[0]

def plot_poincare(ax,section,ts,orbit,
                  s = 1):
    '''A function to plot the recurrent points '''
    first_point = True
    for t,point in section.intersections(ts,orbit):
        ax.scatter(point[0],point[1],point[2],
                   c      = 'xkcd:green',
                   s      = s,
                   label  = r'Poincar\'e return' if first_point else None,
                   marker = 'o')
        first_point = False

def parse_args():
    '''Parse command line parameters'''
    parser  = ArgumentParser(description = __doc__)
    parser.add_argument('--plot',
                        nargs = '*',
                        choices = ['all', 'projections', 'orbit', 'sections', 'map', 'cycles'],
                        default = ['orbit'])
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = DynamicsFactory.products[0])
    parser.add_argument('--fp',
                        type = int,
                        default = 1)
    parser.add_argument('--figs',
                        default = './figs')
    parser.add_argument('--dt',
                        type = float,
                        default = 0.005)
    parser.add_argument('--tFinal',
                        type    = float,
                        default = 100)
    parser.add_argument('--sspTemplate',
                        type    = float,
                        nargs   = 3,
                        default = [1,1,0])
    parser.add_argument('--nTemplate',
                        type   = float,
                        nargs   = 3,
                        default = [1,-1,0])

    return parser.parse_args()

def plot_requested(name,arg):
    '''
    Verify that user has requested a specific plot

    Parameters:
        name                Name of plot
        arg                 List of all plots that user has requested

    Returns True if name appears in arg, or 'all' plots have been specified
    '''
    return len(set(arg).intersection([name,'all']))>0

if __name__ == '__main__':
    rcParams['text.usetex'] = True
    args                    = parse_args()
    dynamics                = DynamicsFactory.create(args)
    integrator              = Integrator(dynamics)
    EQs                     = dynamics.find_equilibria()
    section                 = PoincareSection(dynamics,integrator,
                                              sspTemplate = array(args.sspTemplate),
                                              nTemplate   = array(args.nTemplate))
    x0                      = dynamics.get_start_on_unstable_manifold(EQs[args.fp])
    nstp                    = int(args.tFinal/args.dt)
    ts,orbit                = integrator.integrate(x0, args.tFinal, nstp)
    recurrences             = Recurrences(section,ts,orbit,
                                          filter = lambda point:point[0]>0)    #FIXME

    if plot_requested('orbit',args.plot):
        fig = figure(figsize=(12,12))
        ax1  = fig.add_subplot(111, projection='3d')
        ax1.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                c          = 'xkcd:blue',
                label      = 'Orbit',
                markersize = 1)
        m,_ = EQs.shape
        for i in range(m):
            ax1.scatter(EQs[i,0], EQs[i,1], EQs[i,2],
                       marker = MarkerStyle.filled_markers[i],
                       c      = 'xkcd:red',
                       label  = f'EQ{i}')

        plot_poincare(ax1,section,ts,orbit, s=5)

        suptitle(dynamics.get_title())
        ax1.set_xlabel(dynamics.get_x_label())
        ax1.set_ylabel(dynamics.get_y_label())
        ax1.set_zlabel(dynamics.get_z_label())
        figlegend()
        savefig(join(args.figs,f'{args.dynamics}-orbit'))

    if plot_requested('projections',args.plot):
        fig = figure(figsize=(12,12))
        ax1  = fig.add_subplot(221, projection='3d')
        ax1.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                c          = 'xkcd:blue',
                label      = 'Orbit',
                markersize = 1)
        m,_ = EQs.shape
        for i in range(m):
            ax1.scatter(EQs[i,0], EQs[i,1], EQs[i,2],
                       marker = MarkerStyle.filled_markers[i],
                       c      = 'xkcd:red',
                       label  = f'EQ{i}')

        suptitle(dynamics.get_title())
        ax1.set_xlabel(dynamics.get_x_label())
        ax1.set_ylabel(dynamics.get_y_label())
        ax1.set_zlabel(dynamics.get_z_label())

        ax2  = fig.add_subplot(222)
        ax2.plot(orbit[1,:], orbit[2,:],
                c          = 'xkcd:blue',
                markersize = 1)
        ax2.set_xlabel(dynamics.get_y_label())
        ax2.set_ylabel(dynamics.get_z_label())

        ax3  = fig.add_subplot(223)
        ax3.plot(orbit[0,:], orbit[2,:],
                c          = 'xkcd:blue',
                markersize = 1)
        ax3.set_xlabel(dynamics.get_x_label())
        ax3.set_ylabel(dynamics.get_z_label())

        ax4  = fig.add_subplot(224)
        ax4.plot(orbit[0,:], orbit[1,:],
                c          = 'xkcd:blue',
                markersize = 1)
        ax4.set_xlabel(dynamics.get_x_label())
        ax4.set_ylabel(dynamics.get_y_label())

        figlegend()
        tight_layout()
        savefig(join(args.figs,f'{args.dynamics}-projections'))

    if plot_requested('sections', args.plot):
        fig = figure(figsize=(12,12))
        ax  = fig.gca()
        ax.scatter(recurrences.Section[:, 0], recurrences.Section[:, 1],
                c      = 'xkcd:red',
                marker = 'x',
                s      = 25,
                label  = 'Poincare Section')

        ax.scatter(recurrences.SortedPoincareSection[:, 0], recurrences.SortedPoincareSection[:, 1],
                c      = 'xkcd:blue',
                marker = '+',
                s      = 25,
                label  = 'Sorted Poincare Section')

        ax.scatter(recurrences.InterpolatedPoincareSection[:, 0], recurrences.InterpolatedPoincareSection[:, 1],
                c      = 'xkcd:green',
                marker = 'o',
                s      = 1,
                label  = 'Interpolated Poincare Section')

        ax.set_title('Poincare Section, showing Interpolation')
        ax.set_xlabel('$\\hat{x}\'$')
        ax.set_ylabel('$z$')
        ax.legend()
        savefig(join(args.figs,f'{args.dynamics}-sections'))

    if plot_requested('map',args.plot):
        fig = figure(figsize=(12,12))
        ax = fig.add_subplot(111)
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
                color = 'xkcd:black',
                linestyle = 'dotted',
                label     = '$y=x$')
        ax.legend()
        savefig(join(args.figs,f'{args.dynamics}-map'))

    if plot_requested('cycles',args.plot):
        fig                  = figure(figsize=(12,12))
        sfixed, sspfixed     = recurrences.get_fixed()
        Tnext                = recurrences.getTnext(sspfixed)
        ts_guess,orbit_guess = integrator.integrate(sspfixed, Tnext, nstp)
        print(f'Shortest periodic orbit guessed at: {sspfixed}, Period: {Tnext}')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(orbit_guess[0,:], orbit_guess[1,:], orbit_guess[2,:],
                markersize = 1,
                c          = 'xkcd:blue',
                label      = 'Orbit')
        ax.scatter(orbit_guess[0,0], orbit_guess[1,0], orbit_guess[2,0],
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = f'Start ({orbit_guess[0,0]}, {orbit_guess[1,0]}, {orbit_guess[2,0]})')
        ax.scatter(orbit_guess[0,-1], orbit_guess[1,-1], orbit_guess[2,-1],
                   color  = 'xkcd:red',
                   marker = '+',
                   s      = 64,
                   label  = f'Return ({orbit_guess[0,-1]}, {orbit_guess[1,-1]}, {orbit_guess[2,-1]})')
        # plot_poincare(ax,section,ts,orbit, s=5)
        # try:
            # period, sspfixed =  recurrences.solve(Tnext, sspfixed,
                                                  # integrator = integrator,
                                                  # dynamics   = dynamics)

            # print(f'Shortest periodic orbit is at: {sspfixed}, Period: {period}')

            # _,periodicOrbit        = integrator.integrate(sspfixed, period, nstp)


            # ax.plot(periodicOrbit[0,:], periodicOrbit[1,:], periodicOrbit[2,:],
                    # markersize = 10,
                    # c          = 'xkcd:magenta',
                    # label      = 'periodicOrbit')
        # except:
            # print(exc_info())

        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())
        ax.legend()
        savefig(join(args.figs,f'{args.dynamics}-cycles'))

    show()
