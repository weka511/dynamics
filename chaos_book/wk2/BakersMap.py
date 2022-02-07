# Copyright (C) 2019-2022 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/

'''A contracting baker’s map'''

def f(pt):
    x,y = pt
    return (x/3,2*y) if y<=2  else (x/3+1/2,2*y-1)

def area(u):
    return 0.5*(  u[0][0]*u[1][1] + u[1][0]*u[2][1] + u[2][0]*u[3][1] + u[3][0]*u[0][1]
                - u[1][0]*u[0][1] - u[2][0]*u[1][1] - u[3][0]*u[2][1] - u[0][0]*u[3][1])

u = [(0,0), (1,0), (1,1), (0,1)]

v = [f(pt) for pt in u]

print (area(v)/area(u))
