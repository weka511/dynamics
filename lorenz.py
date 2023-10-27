#!/usr/bin/env python

# Copyright (C) 2017-2023 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import rki
import utilities

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def lorenz(y,sigma=10,b=8/3,rho=28):
    dx = sigma*(y[1]-y[0])
    dy = rho*y[0]-y[1]-y[0]*y[2]
    dz = y[0]*y[1]-b*y[2]
    return [dx,dy,dz]

if __name__=='__main__':
    sigma = 10
    b = 8/3
    rho = 28
    rk = rki.ImplicitRungeKutta2(lambda y: lorenz(y,sigma,b,rho),10,0.000000001)
    driver = rki.Driver(rk,0.000000001,0.5,1.0,0.000000001)

    try:
        nn = 10000
        fig = figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        fig.suptitle(r'Lorenz Equation: {0} iterations'.format(nn))
        ax.set_title(r'$\dot x=\sigma(y-z)\dot y=\rho x -y - xz, \dot z = xy-bz,\sigma={0},b={1},\rho={2}$'.format(rho,b,sigma))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        for y in [utilities.direct_sphere(R=10) for i in range(25)]:
            label = '({0:.2f},{1:.2f},{2:.2f})'.format(y[0],y[1],y[2])
            xs = []
            ys = []
            zs=[]
            for i in range(nn):
                y = driver.step(y)
                xs.append(y[0])
                ys.append(y[1])
                zs.append(y[2])
            ax.plot(xs,ys,zs,label=label)

        ax.legend(loc='best')
        fig.savefig('lorenz.png')
        show()
    except rki.ImplicitRungeKutta.Failed as e:
        print ("caught!",e)
