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

import random,math

# Generate uniform random vector inside sphere, using algorithm from 
# Statistical Mechanics: Algorithms and Computations by Werner Krauth
#
# Parameters:
#     d       Dimensionality
#     sigma   Standard deviation
#     R       Radius

def direct_sphere(d=3,sigma=1,R=1):
    xs      = [random.gauss(0,sigma) for k in range(d)]
    Sigma   = sum([x*x for x in xs])
    upsilon = random.uniform(0,1)**(1/d)
    return [R * upsilon * x/math.sqrt(Sigma) for x in xs]

if __name__=='__main__':
    for i in range(25):
        print (direct_sphere())
        