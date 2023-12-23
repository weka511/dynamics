#!/usr/bin/env python

#   Copyright (C) 2022-2023 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
    Exercise 11.5 Proto-Lorentz system
'''
import numpy  as np
from matplotlib.pyplot import figure, show
from scipy.integrate   import solve_ivp

sigma = 10
beta  = 8/3
rho   = 28

def Velocity(t, stateVec):
    u = stateVec[0]
    v = stateVec[1]
    z = stateVec[2]
    N = np.sqrt(u**2 + v**2)

    return np.array(
        [
            -(sigma+1)* u + (sigma-rho)*v + (1-sigma)*N + v*z,
            (rho-sigma)*u - (sigma+1)*v + (rho+sigma)*N - u*z -u*N,
            v/2 - beta*z
            ],
        float)

if __name__=='__main__':
    tInitial = 0
    tFinal   = 100

    sspSolution = solve_ivp(Velocity, (tInitial, tFinal), np.array([10, 0.0, 3], float) )
    ut = sspSolution.y[0, :]
    vt = sspSolution.y[1, :]
    zt = sspSolution.y[2, :]

    fig = figure(figsize=(12,12))
    ax  = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(ut, vt, zt, color='xkcd:purple', s=1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('z')
    ax.set_title('Miranda & Stone proto-Lorentz')
    show()
