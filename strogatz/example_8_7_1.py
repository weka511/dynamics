#!/usr/bin/env python

# Copyright (C) 2017-2019 Greenweaves Software Limited

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

import sys
sys.path.append('../')
import  matplotlib.pyplot as plt,matplotlib.colors as colors,math,numpy as np,rk4

def f(y):
    r,theta = y
    return (r*(1-r)*(1+r),1)

def plot(h = 0.1,start=(0.1,0),n=10,color='b'):
    pts = [start]
    for i in range(int(2*n*math.pi/h)+1):
        pts.append(rk4.rk4(h,pts[-1],f))
    plt.scatter([r*math.cos(theta) for (r,theta) in pts],
                [r*math.sin(theta) for (r,theta) in pts],
                c=color,
                s=1)

if __name__=='__main__':
    plot()
    plot(start=(2,0),color='r')
    plt.show()
