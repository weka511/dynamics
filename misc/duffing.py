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

from matplotlib import rc
from matplotlib.pyplot import figure, show
import rki

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def duffing(y):
    dx = y[1]
    dy = -0.15*y[1] + y[0] - y[0]**3
    return [dx,dy]

if __name__=='__main__':
    rk = rki.ImplicitRungeKutta2(duffing,10,0.000000001)
    driver = rki.Driver(rk,0.000000001,0.5,1.0,0.000000001)
    try:
        nn = 1000
        fig = figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        fig.suptitle("Duffing's Equation: {0} iterations".format(nn))
        ax.set_title(r'$\dot x = y, \dot y = -0.15 y + x - x^3$')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        for y in [[0,1],[1,1],[2,2],[3,1],[5,5]]:
            label = '({0},{1})'.format(y[0],y[1])
            xs = []
            ys = []
            for i in range(nn):
                y = driver.step(y)
                xs.append(y[0])
                ys.append(y[1])
            ax.plot(xs,ys,label=label)

        ax.legend()
        fig.savefig('duffing.png')
        show()
    except rki.ImplicitRungeKutta.Failed as e:
        print ("caught!",e)
