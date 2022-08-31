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
from contextlib             import AbstractContextManager
from dynamics               import DynamicsFactory
from matplotlib.markers     import MarkerStyle
from matplotlib.pyplot      import figlegend, figure, rcParams, savefig, show, suptitle, tight_layout
from numpy                  import append, arange, argmin, argsort, argwhere, array, cos, cross, dot, identity, \
                                   iinfo, int64, linspace, pi, real, reshape, sin, size, sqrt, stack, zeros
from numpy.linalg           import eig, inv, norm
from os.path                import join, basename, split, splitext
from pathlib                import Path
from scipy.integrate        import solve_ivp
from scipy.interpolate      import splev, splprep, splrep
from scipy.linalg           import eig, norm
from scipy.optimize         import fsolve
from scipy.spatial.distance import pdist, squareform
from sys                    import exc_info
from time                   import time



class Integrator:
    '''This class is used to integrate  the ODEs for a specified Dynamics'''
    def __init__(self,dynamics,
                method = 'LSODA'): # See https://danielmuellerkomorowska.com/2021/02/16/differential-equations-with-scipy-odeint-or-solve_ivp/
        self.dynamics = dynamics
        self.method   = method

    def integrate(self,y0, dt, nstp=1):
        '''
        The integrator of the Lorentz system.
        init_x: the intial condition
        dt : time step
        nstp: number of integration steps.

        return : a [ nstp x 3 ] vector
        '''

        solution = solve_ivp(dynamics.Velocity,  (0.0, dt), y0,
                             method = self.method,
                             t_eval = arange(0.0, dt, dt/nstp))

        if solution.status==0:
            return solution.t, solution.y
        else:
            raise(Exception(f'Integration error: {solution.status} {solution.message}'))

    def Flow(self,deltat,y):                         #FIXME (may need work, as untested with LSODA)
        '''Used to integrate for a single step'''
        solution = solve_ivp(dynamics.Velocity, (0, deltat), y,method=self.method)
        return solution.t[1],solution.y[:,1]

    def Jacobian(self,ssp, t):      #FIXME (may need work, as untested with LSODA)
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
    def __init__(self,dynamics,integrator,
                 sspTemplate = array([1,1,0]),
                 nTemplate   = array([1,-1,0])):
        self.sspTemplate  = sspTemplate/norm(sspTemplate)
        self.nTemplate    = nTemplate/norm(nTemplate)
        self.ProjPoincare = array([self.sspTemplate,
                                   cross(self.sspTemplate,self.nTemplate),
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

    def find_intersection(self,dt, y):
        '''Refine an estimated intersection point '''
        dt_intersection = fsolve(lambda t: self.U(self.Flow(y, t)), dt)[0]
        return self.integrator.Flow(dt_intersection, y)

    def intersections(self,ts, orbit):
        '''Used to iterate through intersections between orbit and section'''
        _,n = orbit.shape
        for i in range(n-1):
            u0 = self.U(orbit[:,i])
            u1 = self.U(orbit[:,i+1])
            if u0<0 and u1>0:
                ratio = (-u0)/((-u0)+u1)
                yield self.find_intersection(ratio*(ts[i+1] - ts[i]), orbit[:,i])

class Recurrences:
    '''This class keeps track of the recurrences of the Poincare map'''
    def __init__(self,section,ts,orbit,
                 filter        = lambda point: True,
                 spline_degree = 5):   # Final plot looks wonky with default in splrep
        self.section       = section
        self.filter        = filter
        self.spline_degree = spline_degree
        self.intersections = [point for _,point in section.intersections(ts,orbit) if filter(point)]
        self.build2D()

    def build2D(self,num  = 1000):
        '''
        Build a 2D view of intersections with Poincare Section
        '''
        n = len(self.intersections)

        if n==0: raise Exception('Orbit does not intersect Poincare Section')

        self.Section = section.project_to_section(array(self.intersections))
        Distance     = squareform(pdist(self.Section))

        # Sort Poincare Section by distance from centre

        self.SortedPoincareSection = self.Section.copy()

        for k in range(n - 1):
            m                                    = argmin(Distance[k, k + 1:]) + k + 1
            saved_poincare_row                   = self.SortedPoincareSection[k + 1, :].copy()
            self.SortedPoincareSection[k + 1, :] = self.SortedPoincareSection[m, :]
            self.SortedPoincareSection[m, :]     = saved_poincare_row

            saved_column                         = Distance[:, k + 1].copy()
            Distance[:, k + 1]                   = Distance[:, m]
            Distance[:, m]                       = saved_column

            saved_row                            = Distance[k + 1, :].copy()
            Distance[k + 1, :]                   = Distance[m, :]
            Distance[m, :]                       = saved_row

        ArcLengthsAfterSorting = zeros(n)  # arclengths of the Poincare section points ordered by distance from centre
        sn                     = zeros(n)  # arclengths of the Poincare section points in dynamical order

        for k in range(n - 1):
            ArcLengthsAfterSorting[k + 1]          = ArcLengthsAfterSorting[k] + Distance[k, k + 1]
            index_in_original_poincare_section     = argwhere(self.Section[:, 0] == self.SortedPoincareSection[k + 1, 0])
            sn[index_in_original_poincare_section] = ArcLengthsAfterSorting[k + 1]

        #Parametric spline interpolation to the Poincare section:
        self.tckPoincare, u              = splprep([self.SortedPoincareSection[:, 0], self.SortedPoincareSection[:, 1]],
                                                   u = ArcLengthsAfterSorting,
                                                   s = 0)
        self.sArray                      = linspace(min(ArcLengthsAfterSorting), max(ArcLengthsAfterSorting),
                                                    num = num)
        self.InterpolatedPoincareSection = self.fPoincare(self.sArray)
        sn1                              = sn[0:-1]
        sn2                              = sn[1:]
        isort                            = argsort(sn1)
        self.sn1                         = sn1[isort]
        self.sn2                         = sn2[isort]
        self.tckReturn                   = splrep(self.sn1,self.sn2,
                                                  k = self.spline_degree)
        self.snPlus1                     = splev(self.sArray, self.tckReturn)

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
        '''Determine time when orbit returns
           This is a hack!!!
        '''
        ssp = sspfixed.copy()
        expectation = self.filter(ssp)
        t   = 0
        for i in range(100):
            print (f'getTnext iteration {i+1}')
            dt = fsolve(lambda dt: self.section.U(integrator.integrate(ssp,dt)[0]),
                       args.tFinal / size(self.Section, 0))[0]
            _,ssp0 = integrator.integrate(ssp,dt,nstp=10)
            ssp = ssp0[:,-1]
            t += dt
            if expectation==self.filter(ssp):
                return t


def parse_args():
    '''Parse command line parameters'''
    parser  = ArgumentParser(description = __doc__)
    parser.add_argument('--plot',
                        nargs = '*',
                        choices = ['all',  'orbit', 'sections', 'map', 'cycles'],
                        default = ['orbit'])
    parser.add_argument('--dynamics',
                        choices = DynamicsFactory.products,
                        default = DynamicsFactory.products[0])
    parser.add_argument('--fp',
                        type    = int,
                        default = 1)
    parser.add_argument('--figs',
                        default = './figs')
    parser.add_argument('--dt',
                        type    = float,
                        default = 0.005)
    parser.add_argument('--s0',
                        type    = float,
                        default = 15.0)
    parser.add_argument('--tFinal',
                        type    = float,
                        default = 100.0)
    parser.add_argument('--sspTemplate',
                        type    = float,
                        nargs   = 3,
                        default = [1,1,0])
    parser.add_argument('--nTemplate',
                        type   = float,
                        nargs   = 3,
                        default = [1,-1,0])
    parser.add_argument('--refine',
                        default = False,
                        action = 'store_true')
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

class Figure(AbstractContextManager):
    '''Context manager for plotting a figure'''
    def __init__(self,
                 figs     = './',
                 name     = '',
                 dynamics = 'Lorentz'):
        self.figs     = figs
        self.name     = name
        self.dynamics = dynamics

    def __enter__(self):
        self.fig = figure(figsize=(12,12))
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type==None and exc_val==None and exc_tb==None:
            savefig(join(self.figs,f'{splitext(basename(__file__))[0]}-{self.dynamics}-{self.name}'))
            return True
        else:
            return False

class Timer(AbstractContextManager):
    '''
    Context Manager for estimating time
    Prints the elapsed time from __enter__(...) to __exit__(...)
    '''
    def __init__(self,name='Timer'):
        self.name = name

    def __enter__(self):
        self.start = time()
        return self.start

    def __exit__(self,exc_type, exc_val, exc_tb):
        print (f'{self.name}: Elapsed time = {time()-self.start:.0f} seconds')
        return exc_type==None and exc_val==None and exc_tb==None

if __name__ == '__main__':
    with Timer():
        rcParams['text.usetex'] = True
        args                    = parse_args()
        dynamics                = DynamicsFactory.create(args)
        integrator              = Integrator(dynamics,method='RK45')
        EQs                     = dynamics.find_equilibria()
        section                 = PoincareSection(dynamics,integrator,
                                                  sspTemplate = array(args.sspTemplate),
                                                  nTemplate   = array(args.nTemplate))
        y0                      = dynamics.get_start_on_unstable_manifold(EQs[args.fp])
        ts,orbit                = integrator.integrate(y0, args.tFinal,int(args.tFinal/args.dt))
        recurrences             = Recurrences(section,ts,orbit,
                                              filter = lambda point:point[0]>0)
        sfixed,psfixed,sspfixed = recurrences.get_fixed(s0 = args.s0)
        Tnext                   = recurrences.getTnext(sspfixed)

        if plot_requested('orbit',args.plot):
            with Figure(figs     = args.figs,
                        dynamics = dynamics.name,
                        name     = 'orbit') as fig:
                ax = fig.add_subplot(111, projection='3d')
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
                ax.scatter(orbit[0,-1], orbit[1,-1], orbit[2,-1],
                               marker = MarkerStyle.filled_markers[-1],
                               c      = 'xkcd:red',
                               label  = f'T={ts[-1]:.3f}')
                for index,point in enumerate(recurrences.intersections):
                    ax.scatter(point[0],point[1],point[2],
                               c      = 'xkcd:green',
                               s      = 5,
                               label  = r'Poincar\'e return' if index==0 else None,
                               marker = 'o')
                suptitle(dynamics.get_title())
                ax.set_xlabel(dynamics.get_x_label())
                ax.set_ylabel(dynamics.get_y_label())
                ax.set_zlabel(dynamics.get_z_label())
                ax.legend()

        if plot_requested('sections',args.plot):
            with Figure(figs     = args.figs,
                        dynamics = dynamics.name,
                        name     = 'sections') as fig:
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
                ax.scatter(psfixed[0], psfixed[1],
                           c      = 'xkcd:magenta',
                           marker = 'D',
                           s      = 25,
                           label  = 'psfixed')
                ax.set_title('Poincare Section, showing Interpolation')
                ax.set_xlabel('$\\hat{x}\'$')
                ax.set_ylabel('$z$')
                ax.legend()

        if plot_requested('map',args.plot):
            with Figure(figs     = args.figs,
                        dynamics = dynamics.name,
                        name     = 'map') as fig:
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
                        color     = 'xkcd:black',
                        linestyle = 'dotted',
                        label     = '$y=x$')
                ax.axvline(x = sfixed,
                           c     = 'xkcd:purple',
                           lw    = 1.0,
                           ls    = '--',
                           label = f'sfixed={sfixed:.4f}')
                ax.legend()
                ax.set_title('Arc Lengths')


        if plot_requested('cycles',args.plot):
            with Figure(figs     = args.figs,
                        dynamics = dynamics.name,
                        name     = 'cycles') as fig:

                ts_guess,orbit_guess = integrator.integrate(sspfixed, Tnext, nstp=100)
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
                # for index,point in enumerate(intersections):
                    # ax.scatter(point[0],point[1],point[2],
                               # c      = 'xkcd:green',
                               # s      = 5,
                               # label  = r'Poincar\'e return' if index==0 else None,
                               # marker = 'o')
                if args.refine:
                    period, sspfixed =  recurrences.solve(Tnext, sspfixed,
                                                          integrator = integrator,
                                                          dynamics   = dynamics)

                    print(f'Shortest periodic orbit is at: {sspfixed}, Period: {period}')

                    _,periodicOrbit        = integrator.integrate(sspfixed, period, nstp=100)


                    ax.plot(periodicOrbit[0,:], periodicOrbit[1,:], periodicOrbit[2,:],
                            markersize = 10,
                            c          = 'xkcd:magenta',
                            label      = 'periodicOrbit')


                ax.set_xlabel(dynamics.get_x_label())
                ax.set_ylabel(dynamics.get_y_label())
                ax.set_zlabel(dynamics.get_z_label())
                ax.legend()
    show()