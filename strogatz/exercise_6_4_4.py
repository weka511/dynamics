#!/usr/bin/env python

# Copyright (C) 2019-2023 Simon Crase

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
    Exercise 6.4.4 from Strogatz
    Plot phase portraits for a number of ODEs
'''

from  matplotlib.pyplot import figure,show
from phase import generate,plot_phase_portrait,plot_stability,right_upper_quadrant

def f(x,y):
    return (x*(1-y),y*(rho-x))

for rho in [0.1,0.25,
            0.5,1.0,2.0
            ]:
    fig = figure(figsize=(20,20))
    ax = fig.add_subplot(1,1,1)
    X,Y,U,V,fixed_points=generate(f=f,nx=256,ny=256,xmin=0,xmax=3.5,ymin=0,ymax=3.5)
    plot_phase_portrait(X,Y,U,V,fixed_points,
                        title=r'$\dot{{x}}=x(1-y),\dot{{y}}=y(\rho-x);\rho={0}$'.format(rho),
                        ax=ax)
    plot_stability(f=f,fixed_points=fixed_points,Limit=5,step=0.1,N=5000,accept=right_upper_quadrant,K=10,ax=ax)
    fig.suptitle('Example 6.4.4')

show()
