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

'''A collection of classes that model ODEs '''

from abc             import ABC,abstractmethod
from numpy           import arange, array, dot, exp, identity, imag, isreal, pi, real, reshape, size, sqrt, stack, zeros
from scipy.linalg    import eig, norm
from scipy.integrate import solve_ivp

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
        '''
        Initial condition as a slight perturbation to specified fixed point
        in the direction of eigenvector with largest eigenvalue
        '''
        Aeq0 = self.StabilityMatrix(eq0)
        _,v  = eig(Aeq0)
        v1   = real(v[:, 0])
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

    def find_equilibria(self,epsilon=0.002):
        '''See Chaosbook version 15.2, Jan 26 2017, equation (2.28)'''
        vector = array([self.c, -self.c/self.a, self.c/self.a])
        term2  = 0.5*sqrt(1-4*self.b*self.a/self.c**2)
        if self.a*self.b/self.c**2<epsilon:
            return stack([( 0.5 + term2)*vector,
                          array([self.a*self.b/self.c, -self.b/self.c, self.b/self.c])])
        else:
            return stack([( 0.5 + term2)*vector,(0.5 - term2)*vector])

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

class Equilibrium:
    '''This class represents one equilibrium point'''
    @classmethod
    def create(self,dynamics):
        '''Create set of equilibria for specified Dynamics'''
        eqs = dynamics.find_equilibria()
        m,_ = eqs.shape
        return [Equilibrium(dynamics,eqs[i,:]) for i in range(m)]

    def __init__(self,dynamics,eq):
        self.dynamics = dynamics
        self.eq       = eq.copy()
        self.w,self.v = eig(self.dynamics.StabilityMatrix(self.eq))

    def get_eigendirections(self):
        for i in range(len(self.w)):
            yield self.w[i], self.v[:,i]

    def description(self):
        yield f'Eq: ({self.eq[0]:.4},{self.eq[1]:.4},{self.eq[2]:.4})'
        for i in range(len(self.w)):
            if isreal(self.w[i]):
                yield f'{real(self.w[i]):.4f}'
            elif imag(self.w[i])>0:
                T      = 2*pi/imag(self.w[i])
                Lambda = exp(real(self.w[i])*T)
                yield f'Period={T:.4f}, Lambda={Lambda:.4f}'

class Orbit:
    '''Represents the orbit, starting at a specfied point '''
    @classmethod
    def get_start(cls,
                  epsilon     = 0.00001,
                  direction   = array([1,1,1]),
                  orientation = +1,
                  origin      = array([0,0,0])):
        return origin.eq + orientation*epsilon*direction

    def __init__(self,
                 dynamics,
                 dt          = 10.0,
                 nstp        = 10000,
                 epsilon     = 0.00001,
                 direction   = array([1,1,1]),
                 orientation = +1,
                 origin      = array([0,0,0]),
                 eigenvalue  = 1,
                 method      = 'RK45',
                 events      = None):
        '''
        Parameters:
            dynamics,
            dt
            nstp
            epsilon
            direction
            orientation
            origin
            eigenvalue
            method
            events
        '''

        self.dynamics    = dynamics
        self.direction   = direction
        self.eigenvalue  = eigenvalue
        self.orientation = orientation
        self.method      = method
        y0               = Orbit.get_start(epsilon     = epsilon,
                                           direction   = direction,
                                           orientation = orientation,
                                           origin      = origin)
        solution         = solve_ivp(dynamics.Velocity,  (0.0, dt), y0,
                                     method = self.method,
                                     t_eval = arange(0.0, dt, dt/nstp),
                                     events = events)
        if solution.status==0:
            self.orbit    = solution.y
            self.t        = solution.t
            self.nfev     = solution.nfev
            self.t_events = solution.t_events
            self.y_events = solution.y_events
        else:
            raise Exception(f'solve_ivp failed {solution.status}')

    def __len__(self):
        '''Number of time points in solution'''
        return len(self.t)

    def Flow(self,deltat,y,
             nstp = 1):
        '''Used to integrate for a part of orbit'''
        solution = solve_ivp(self.dynamics.Velocity, (0, deltat), y,
                             method = self.method,
                             t_eval = arange(0.0, deltat, deltat/nstp) if nstp>1 else None)
        return  (solution.t, solution.y) if nstp>1 else (solution.t[-1],solution.y[:,-1])

    def Flow1(self,dt,y0,
              epsilon = 1.0e-12,
              t_eval  = None,
              events  = None):
        return solve_ivp(self.dynamics.Velocity,  (0.0, dt), y0+epsilon*self.dynamics.Velocity(0,y0),
                         method = self.method,
                         t_eval = t_eval,
                         events = events)


    def Jacobian(self,t, ssp):
        '''
        Jacobian function for the trajectory started on ssp, evolved for time t

        Inputs:
            ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
            t: Integration time
        Outputs:
            J: Jacobian of trajectory f^t(ssp). dxd NumPy array
        '''

        Jacobian0                       = identity(self.dynamics.d)
        sspJacobian0                    = zeros(self.dynamics.d + self.dynamics.d ** 2)
        sspJacobian0[0:self.dynamics.d] = ssp
        sspJacobian0[self.dynamics.d:]  = reshape(Jacobian0, self.dynamics.d ** 2)
        solution                        = solve_ivp(self.dynamics.JacobianVelocity, (0, t), sspJacobian0,
                                                    method=self.method)
        sspJacobianSolution             = solution.y[:,-1]
        return sspJacobianSolution[ self.dynamics.d:].reshape((self.dynamics.d, self.dynamics.d))

    def get_events(self,n = 0):
        '''Used to find points where orbit crosses section'''
        for t,y in zip(self.t_events[n],self.y_events[n]):
            yield t,array(y)
