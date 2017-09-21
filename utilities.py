# Copyright (C) 2017 Greenweaves Software Pty Ltd

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

def direct_sphere(d=3,sigma=1,R=1):
    samples=[random.gauss(0,sigma) for k in range(d)]
    Sigma=sum([x*x for x in samples])
    upsilon=random.uniform(0,1)**(1/d)
    return [R*upsilon*x/math.sqrt(Sigma) for x in samples]
