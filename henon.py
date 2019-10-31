# Copyright (C) 2019 Greenweaves Software Limited

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

def henon(x,y,a,b):
    return 1-a*x*x+y,b*x

if __name__=='__main__':
    import argparse, matplotlib.pyplot as plt
    parser = argparse.ArgumentParser('Henon mapping')
    parser.add_argument('-a','--a',type=float,default=1.4)
    parser.add_argument('-b','--b',type=float,default=0.3)
    parser.add_argument('-N','--N',type=int,default=100000)
    args = parser.parse_args()

    xs   = []
    ys   = []
    x    = 1
    y    = 1
    for i in range(args.N):
        x,y = henon(x,y,args.a,args.b)
        xs.append(x)
        ys.append(y)
    plt.scatter(xs,ys,s=1,marker='.')
    plt.show()