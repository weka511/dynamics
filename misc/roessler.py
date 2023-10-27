#!/usr/bin/env python
# Copyright (C) 2017-2013 Simon Crase

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

def roessler(y, a = 0.2, b = 0.2, c = 5.7):
    dx = -y[1] - y[2]
    dy = y[0] + a*y[1]
    dz = b + y[2]*(y[0]-c)
    return [dx,dy,dz]

if __name__=='__main__':
    a = 0.2
    b = 0.2
    c = 5.7
    rk = rki.ImplicitRungeKutta2(lambda y: roessler(y,a,b,c),10,0.000000001)
    driver = rki.Driver(rk,0.000000001,0.5,1.0,0.000000001)

    try:
        nn = 5000
        fig = figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1,projection='3d')
        fig.suptitle(r'R\"ossler Equation: {0} iterations'.format(nn))
        ax.set_title(r'$\dot x=-y-z,\dot y=x+ay,\dot z = b + z(x-c),a={0},b={1},c={2}$'.format(a,b,c))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        for y in [utilities.direct_sphere(R=2) for i in range(25)]:
            label = '({0:.2f},{1:.2f},{2:.2f})'.format(y[0],y[1],y[2])
            xs = []
            ys = []
            zs = []
            for i in range(nn):
                y = driver.step(y)
                xs.append(y[0])
                ys.append(y[1])
                zs.append(y[2])
            ax.plot(xs,ys,zs,label=label)

        ax.legend()
        fig.savefig('roessler.png')
        show()
    except rki.ImplicitRungeKutta.Failed as e:
        print ("caught!",e)
