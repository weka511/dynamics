#!/usr/bin/env python

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

'''Kermack & McKendrick model of an epidemic'''
from matplotlib.pyplot import figure, grid, legend, plot,  show,  title
from math              import exp


aa      = [1,2,3]
bs      = [0.1,0.2,0.3]
h       = 0.1
l       = 10.0
colours = ['xkcd:purple',
           'xkcd:green',
           'xkcd:blue',
           'xkcd:pink',
           'xkcd:brown',
           'xkcd:red',
           'xkcd:light blue',
           'xkcd:teal',
           'xkcd:orange'
           ]


us      = [h*u for u in range(0,int(l/h))]
u2s     = [exp(-u) for u in us]

figure(figsize = (10,10))

plot(us,u2s,
     c     = 'xkcd:black',
     label = r'$e^{-u}$')

for i,a in enumerate(aa):
    for j,b in enumerate(bs):
        u1s = [a - b *u for u in us]
        plot(us,u1s,
             c     = colours[len(bs)*i+j],
             label = 'a={0},b={1}'.format(a,b))


title('Kermack & McKendrick model of an epidemic')
grid(True)
legend()
show()
