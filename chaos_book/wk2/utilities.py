# Copyright (C) 2017-2022 Greenweaves Software Limited

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

from math                 import sqrt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot    import figure, show
from random               import gauss, uniform

# direct_sphere
#
# Generate uniform random vector inside sphere, using algorithm from
# Statistical Mechanics: Algorithms and Computations by Werner Krauth
#
# Parameters:
#     d       Dimensionality
#     sigma   Standard deviation
#     R       Radius

def direct_sphere(d=3,sigma=1,R=1):
    xs      = [gauss(0,sigma) for k in range(d)]
    Sigma   = sum([x*x for x in xs])
    upsilon = uniform(0,1)**(1/d)
    return [R * upsilon * x/sqrt(Sigma) for x in xs]

# direct_surface
#
# Generate uniform random vector on surface of sphere, using algorithm from
# Statistical Mechanics: Algorithms and Computations by Werner Krauth
#
# Parameters:
#     d       Dimensionality
def direct_surface(d=3):
    sigma = 1/sqrt(d)
    xs    = [gauss(0,sigma) for k in range(d)]
    Sigma = sum([x*x for x in xs])
    return [x/sqrt(Sigma) for x in xs]

if __name__=='__main__':

    fig = figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xs=[]
    ys=[]
    zs=[]
    for i in range(1000):
        pt = direct_sphere()

        xs.append(pt[0])
        ys.append(pt[1])
        zs.append(pt[2])
    ax1.scatter(xs,ys,zs,color='b')

    xs=[]
    ys=[]
    zs=[]
    for i in range(1000):
        pt = direct_surface()
        xs.append(pt[0])
        ys.append(pt[1])
        zs.append(pt[2])
    ax1.scatter(xs,ys,zs,color='r',alpha=0.5)
    show()
