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

import argparse
import matplotlib.pyplot as plt

def henon(x,y,a=1.4,b=0.3):
    return 1 - a*x*x+y,b*x

def generate_henon(N         = 10000,
                   transient = 1000,
                   a         = 1.4,
                   b         = 0.3,
                   x_min     = 0,
                   x_max     = 0.5,
                   y_min     = 0.2,
                   y_max     = 0.3,
                   x0        = 0,
                   y0        = 0):

    xs   = []
    ys   = []
    x    = x0
    y    = y0
    for i in range(args.N):
        x,y = henon(x,y,a=a,b=b)
        if i>args.transient and x_min<x and x<x_max and y_min<y and y<y_max:
            xs.append(x)
            ys.append(y)
    return xs,ys

if __name__=='__main__':

    parser = argparse.ArgumentParser('Henon mapping')
    parser.add_argument('-a','--a',type=float,default=1.4,help='Parameter for Henon mapping')
    parser.add_argument('-b','--b',type=float,default=0.3,help='Parameter for Henon mapping')
    parser.add_argument('-N','--N',type=int,default=100000,help='Number of points')
    parser.add_argument('-T','--transient',type=int,default=10000,help='Number of points to skip at beginning')
    parser.add_argument('--show',default=False,action='store_true',help='Show plot')
    parser.add_argument('--xmin',type=float,default=0.,help='Parameter for Henon mapping')
    parser.add_argument('--xmax',type=float,default=1,help='Parameter for Henon mapping')
    parser.add_argument('--ymin',type=float,default=0,help='Parameter for Henon mapping')
    parser.add_argument('--ymax',type=float,default=1,help='Parameter for Henon mapping')
    args = parser.parse_args()

    xs,ys = generate_henon(N=args.N,transient=args.transient,a=args.a,b=args.b,
                           x_min=args.xmin,x_max=args.xmax,y_min=args.ymin,y_max=args.ymax)
    plt.figure(figsize=(20,20))
    plt.scatter(xs,ys,s=1,marker='.')
    plt.xlim(args.xmin,args.xmax)
    plt.ylim(args.ymin,args.ymax)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('henon.png')
    if args.show:
        plt.show()
