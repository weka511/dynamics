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

'''
Generate random points in or on sphere, using algorithm from
Statistical Mechanics: Algorithms and Computations by Werner Krauth
'''

import random
import numpy as np
from matplotlib.pyplot import figure, show


def direct_sphere(d = 3,
                  n = 1,
                  sigma = 1,
                  R = 1,
                  rng = np.random.default_rng()):
    '''
    Generate uniform random vector inside sphere

    Parameters:
        d       Dimensionality
        n       Number of points
        sigma   Standard deviation
        R       Radius
        rng     Random number generator
    '''
    x = rng.normal(loc=0,scale=sigma,size=(n,d))
    Sigma = (x**2).sum(axis=-1)
    upsilon = rng.uniform(0,1,size=n)**(1/d)
    return R * np.expand_dims(upsilon/np.sqrt(Sigma),axis=1) * x


def direct_surface(d = 3,
                   n = 1,
                   rng = np.random.default_rng()):
    '''
     Generate uniform random vector on surface of sphere

    Parameters:
        d       Dimensionality
        n       Number of points
        rng     Random number generator
    '''
    sigma = 1/np.sqrt(d)
    x = rng.normal(loc=0,scale=sigma,size=(n,d))
    Sigma =  (x**2).sum(axis=-1)
    return x/np.expand_dims(np.sqrt(Sigma),axis=1)

if __name__=='__main__':
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')

    pt1 = direct_sphere(n=1000)
    ax.scatter(pt1[:,0],pt1[:,1],pt1[:,2],color='xkcd:blue', label='Sphere')

    pt2 = direct_surface(n=100)
    ax.scatter(pt2[:,0],pt2[:,1],pt2[:,2],color='xkcd:red',alpha=0.5, label='Surface')
    ax.legend()
    show()
