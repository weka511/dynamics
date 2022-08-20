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

'''Replicate Figure 4.5'''

from Lorentz           import Rossler
from matplotlib.pyplot import figure, plot, show, tight_layout
from numpy             import arange, exp, imag, isreal, pi, real
from numpy.linalg      import eig
from scipy.integrate   import solve_ivp

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

    def describe(self):
        print (f'{self.eq}')
        w,v = eig(self.dynamics.StabilityMatrix(self.eq))
        for i in range(len(w)):
            if isreal(w[i]):
                print (f'{real(w[i])} {exp( real(w[i]))}')
            elif imag(w[i])>0:
                T = 2*pi/imag(w[i])
                Lambda = exp(real(w[i])*T)
                print (f'Period={T}, Lambda={Lambda}')

def get_orbit(eqs,
              dt       = 100.0,
              nstp     = 10000,
              epsilon  = 0.0001,
              sign     = +1,
              origin   = 0):

    return solve_ivp(dynamics.Velocity,  (0.0, dt), eqs[origin].eq + sign*epsilon*(eqs[origin].eq - eqs[1-origin].eq),
                         method = 'RK45',
                         t_eval = arange(0.0, dt, dt/nstp)).y


if __name__=='__main__':
    dynamics    = Rossler()
    eqs         = Equilibrium.create(dynamics)
    for eq in eqs:
        eq.describe()

    orbit_plus  = get_orbit(eqs, dt=50)
    orbit_minus = get_orbit(eqs, sign = -1)

    fig = figure(figsize=(12,12))
    ax  = fig.add_subplot(221, projection='3d')

    ax.plot(orbit_plus[0,:],orbit_plus[1,:],orbit_plus[2,:],
            c          = 'xkcd:blue',
            markersize = 1,
            label      = 'Orbit')
    s1 =ax.scatter(eqs[0].eq[0], eqs[0].eq[1], eqs[0].eq[2],
               c      = 'xkcd:green',
               s      = 25,
               marker = 'X',
               label  = f'({eqs[0].eq[0]:.4f},{eqs[0].eq[1]:.4f},{eqs[0].eq[2]:.4f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax  = fig.add_subplot(222, projection='3d')
    ax.plot(orbit_minus[0,:],orbit_minus[1,:],orbit_minus[2,:],
            c          = 'xkcd:red',
            markersize = 1)
    ax.scatter(eqs[0].eq[0], eqs[0].eq[1], eqs[0].eq[2],
               c      = 'xkcd:green',
               s      = 25,
               marker = 'X')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    orbit_plus  = get_orbit(eqs, origin= 1)
    orbit_minus = get_orbit(eqs, sign = -1, origin= 1)

    ax = fig.add_subplot(223, projection='3d')

    ax.plot(orbit_plus[0,:],orbit_plus[1,:],orbit_plus[2,:],
            c          = 'xkcd:blue',
            markersize = 1)
    s2=ax.scatter(eqs[1].eq[0], eqs[1].eq[1], eqs[1].eq[2],
               c      = 'xkcd:black',
               s      = 25,
               marker = '+',
               label  = f'({eqs[1].eq[0]:.4f},{eqs[1].eq[1]:.4f},{eqs[1].eq[2]:.4f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = fig.add_subplot(224, projection='3d')
    ax.plot(orbit_minus[0,:], orbit_minus[1,:], orbit_minus[2,:],
            c          = 'xkcd:red',
            markersize = 1)
    ax.scatter(eqs[1].eq[0], eqs[1].eq[1], eqs[1].eq[2],
               c      = 'xkcd:black',
               s      = 25,
               marker = '+')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.legend(handles = [s1,s2], title   = 'Equilibria')
    tight_layout()
    show()
