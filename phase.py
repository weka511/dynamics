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

import numpy as np

def ff(x,t,f):
    u,v=f(x[0],x[1])
    return [u]+[v]

def generate(ex,nx=64, ny = 64):
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    X, Y = np.meshgrid(x, y)
    U,V=ex(X,Y)
    return X,Y,U,V

@np.vectorize
def nullclines(u,v):
    def y(v,offset=0):
        return offset if v<0 else offset+1        
    return y(v) if u<0 else y(v,offset=2)

if __name__=='__main__':
    import matplotlib.pyplot as plt,matplotlib.colors as colors
    def ex(x,y):
        return x+np.exp(-y),-y
    
    X,Y,U,V=generate(ex)
    plt.pcolor(X,Y,nullclines(U,V),cmap=plt.cm.inferno)
    plt.streamplot(X, Y, U, V, color=U, linewidth=1, cmap=plt.cm.inferno)
    plt.colorbar()    
    plt.show()