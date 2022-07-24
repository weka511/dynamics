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
from matplotlib.pyplot      import figure, rcParams, savefig, show, suptitle
from numpy                  import append, arange, argmin, argsort, argwhere, array, cos, cross, dot, iinfo, int64, linspace, pi, real, sin, size, sqrt, zeros
from numpy.linalg           import eig, inv, norm
from os.path                import join, basename, split
from pathlib                import Path
from scipy.integrate        import solve_ivp
from scipy.interpolate      import splev, splprep, splrep
from scipy.linalg           import eig, norm
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform

class Dynamics(ABC):

    def __init__(self,name=None):
        self.name = name

    '''This abstract class represents the dynamics, i.e. the differential equation.'''
    @abstractmethod
    def create_eqs(self):
        ...

    @abstractmethod
    def Velocity(self, t,stateVec):
        ...

    @abstractmethod
    def StabilityMatrix(self,stateVec):
        ...

    def get_title(self):
        return fr'{self.name} $\sigma=${self.sigma}, $\rho=${self.rho}, b={self.b}'

    def get_x_label(self):
        return 'x'

    def get_y_label(self):
        return 'y'

    def get_z_label(self):
        return 'z'

    def get_start_on_unstable_manifold(self,eq0, eps=1e-6):
        '''Initial condition as a slight perturbation to the eq0 in v1 direction'''
        Aeq0            = self.StabilityMatrix(eq0)
        _, eigenVectors = eig(Aeq0)
        v1              = real(eigenVectors[:, 0])
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

    def create_eqs(self):
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

        x = stateVec[0]
        y = stateVec[1]
        z = stateVec[2]

        return array([self.sigma * (y-x),
                      self.rho*x - y - x*z,
                      x*y - self.b*z])

    def StabilityMatrix(self,stateVec):
        '''
        return the stability matrix at a state point.
        stateVec: the state vector in the full space. [x, y, z]
        '''

        x = stateVec[0]
        y = stateVec[1]
        z = stateVec[2];
        return array([
            [-self.sigma, self.sigma, 0],
            [self.rho-z,  -1 ,       -x],
            [y,           x,         -self.b]
        ])



class PseudoLorentz(Dynamics):
    '''Dynamics of Pseudo Lorentz Equation'''
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        super().__init__('PseudoLorentz')
        self.sigma = sigma
        self.rho   = rho
        self.b     = b

    def create_eqs(self):
        eq0 = [0,0,0]
        eq1 = [0,2*self.b*(self.rho-1),self.rho-1]
        return  array([eq0,eq1])

    def Velocity(self,t,stateVec):
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

class Rossler(Dynamics):
    def __init__(self,
                 a = 0.2,
                 b = 0.2,
                 c = 5.7):
        super().__init__('Rossler')
        self.a = a
        self.b = b
        self.c = c

    def create_eqs(self):
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

        x = stateVec[0]
        y = stateVec[1]
        z = stateVec[2]

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

        x, y, z = ssp  # Read state space points


        return array([[0, -1, -1],
                      [1, self.a, 0],
                      [z, 0, x-self.c]],
                     float)

    def get_title(self):
        return 'Rossler'

class Integrator:
    '''This class is used to integrate  the ODEs for a specified Dynamics'''
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

        bunch = solve_ivp(dynamics.Velocity, (0, dt), init_x, t_eval = arange(0,dt,dt/nstp))
        if bunch.status==0:
            return bunch.t, bunch.y
        else:
            raise(Exception(f'Integration error: {bunch.status} {bunch.message}'))



    def Flow(self,deltat,y):
        '''Used to integrate for a single step'''
        bunch = solve_ivp(dynamics.Velocity, (0, deltat), y)
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
                 e_z         = array([0, 0, 1], float)):
        e_x               = array([1, 0, 0], float)  # Unit vector in x-direction
        self.sspTemplate  = dot(PoincareSection.zRotation(theta), e_x)#  if len(sspTemplate)==1 else sspTemplate
        self.nTemplate    = dot(PoincareSection.zRotation(pi/2), self.sspTemplate) #if len(nTemplate)==1 else nTemplate
        self.ProjPoincare = array([self.sspTemplate,
                                     e_z,
                                     self.nTemplate], float)
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

    def project_to_section(self,points):
        '''Transform point on the section from (x,y,z) to coordinates embedded in surface'''
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



    def create_arclengths(self,ts,orbit):
        self.Section               = self.project_to_section(array([point for _,point in self.intersections(ts,orbit)]))
        Distance              = squareform(pdist(self.Section))
        self.SortedPoincareSection = self.Section.copy()
        ArcLengths            = zeros(size(self.SortedPoincareSection, 0))
        sn                    = zeros(size(self.Section, 0)) # the arclengths of the Poincare section points keeping their dynamical order for use in the return map
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

            ArcLengths[k + 1] = ArcLengths[k] + Distance[k, k + 1] #Assign the arclength of (k+1)th element:
            #Find this point in the PoincareSection array and assign sn to its  corresponding arclength:
            sn[argwhere(self.Section[:, 0] == self.SortedPoincareSection[k + 1, 0])] = ArcLengths[k + 1]

        #Parametric spline interpolation to the Poincare section:
        self.tckPoincare, u = splprep([self.SortedPoincareSection[:, 0], self.SortedPoincareSection[:, 1]],
                                      u = ArcLengths,
                                      s = 0)
        self.sArray = linspace(min(ArcLengths), max(ArcLengths), 1000)
        #Evaluate the interpolation:
        self.InterpolatedPoincareSection = self.fPoincare(self.sArray)

        sn1            = sn[0:-1]
        sn2            = sn[1:]
        isort          = argsort(sn1)
        self.sn1       = sn1[isort]  # sort radii1
        self.sn2       = sn2[isort]  # sort radii2
        self.tckReturn = splrep(self.sn1,self.sn2)
        self.snPlus1   = splev(self.sArray, self.tckReturn)

    def get_fixed(self, s0=10.0):
        sfixed = fsolve(lambda r: splev(r, self.tckReturn) - r, s0)[0]
        print (sfixed,self.fPoincare(sfixed),self.project_to_space(self.fPoincare(sfixed)))
        return  sfixed,self.project_to_space(self.fPoincare(sfixed))

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
    for t,point in section.intersections(ts,orbit):
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
    parser.add_argument('--plot',
                        nargs = '*',
                        choices = ['all', 'orbit', 'sections', 'map', 'cycles'],
                        default = ['orbit'])
    parser.add_argument('--dynamics',
                        choices=['Lorentz', 'PseudoLorentz', 'Rossler'],
                        default = 'Lorentz')
    parser.add_argument('--fp',
                        type = int,
                        default = 1)
    parser.add_argument('--figs',
                        default = './figs')
    return parser.parse_args()

def create_dynamics(args):
    if args.dynamics == 'Lorentz':
        return Lorentz()
    if args.dynamics == 'PseudoLorentz':
        return PseudoLorentz()
    if args.dynamics == 'Rossler':
        return Rossler()
    raise Exception(f'Unknown dynamics: {args.dynamics}')


def plot_requested(name,arg):
    return len(set(arg).intersection([name,'all']))>0

if __name__ == '__main__':

    rcParams['text.usetex'] = True
    args                    = parse_args()
    dynamics                = create_dynamics(args)
    integrator              = Integrator(dynamics)
    EQs                     = dynamics.create_eqs()
    section                 = PoincareSection(dynamics,integrator,
                                              sspTemplate = None,#EQs[args.fp],
                                              nTemplate   = None)#array([1,0,0]))

    x0                      = dynamics.get_start_on_unstable_manifold(EQs[args.fp])
    dt                      = 0.005
    tFinal                  = 100
    nstp                    = tFinal/dt
    ts,orbit                = integrator.integrate(x0, tFinal, nstp)
    section.create_arclengths(ts,orbit)

    if plot_requested('orbit',args.plot):
        fig = figure(figsize=(12,12))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                c          = 'xkcd:blue',
                label      = 'Orbit',
                markersize = 1)
        m,_ = EQs.shape
        for i in range(m):
            ax.scatter(EQs[i,0], EQs[i,1], EQs[i,2],
                       marker = MarkerStyle.filled_markers[i],
                       c      = 'xkcd:red',
                       label  = f'EQ{i}')

        plot_poincare(ax,section,ts,orbit, s=5)
        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())
        ax.legend()
        savefig(join(args.figs,f'{args.dynamics}-orbit'))

    if plot_requested('sections', args.plot):
        fig = figure(figsize=(12,12))
        ax  = fig.gca()
        ax.scatter(section.Section[:, 0], section.Section[:, 1],
                c      = 'xkcd:red',
                marker = 'x',
                s      = 25,
                label      = 'Poincare Section')

        ax.scatter(section.SortedPoincareSection[:, 0], section.SortedPoincareSection[:, 1],
                c      = 'xkcd:blue',
                marker = '+',
                s      = 25,
                label  = 'Sorted Poincare Section')

        ax.scatter(section.InterpolatedPoincareSection[:, 0], section.InterpolatedPoincareSection[:, 1],
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
        ax.scatter(section.sn1, section.sn2,
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = 'Sorted')
        ax.scatter(section.sArray, section.snPlus1,
                   color  = 'xkcd:blue',
                   marker = 'o',
                   s      = 1,
                   label = 'Interpolated')
        ax.plot(section.sArray, section.sArray,
                color = 'xkcd:black',
                linestyle = 'dotted',
                label     = '$y=x$')
        ax.legend()
        savefig(join(args.figs,f'{args.dynamics}-map'))

    if plot_requested('cycles',args.plot):
        fig = figure(figsize=(12,12))
        sfixed, sspfixed = section.get_fixed()
        # fdeltat = lambda dt: section.U(integrator.integrate(sspfixed,dt)[0])#section.Flow(sspfixed, deltat))
        # Tguess = tFinal / size(section.Section, 0)
        Tnext = fsolve(lambda dt: section.U(integrator.integrate(sspfixed,dt)[0]),
                       tFinal / size(section.Section, 0))#[0]
        ts,orbit      = integrator.integrate(sspfixed, Tnext, nstp)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                markersize = 1,
                c          = 'xkcd:blue',
                label      = 'Orbit')
        ax.scatter(orbit[0,0], orbit[1,0], orbit[2,0],
                   color  = 'xkcd:red',
                   marker = 'x',
                   s      = 64,
                   label  = 'Start')
        ax.legend()
        savefig(join(args.figs,f'{args.dynamics}-cycles'))

    show()
