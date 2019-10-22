#    ex2_integrator.py  Integration example from week 2
#    Copyright (C) 2018  Simon Crase  simon@General

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np,math

class Integrator:
    def __init__(self, xMin, xMax, N,f=lambda x:x*x*math.exp(-x)*math.sin(x)):
        self.xMin=xMin
        self.xMax=xMax
        self.N=N
        self.f=f
            
    def integrate(self):       
        deltaX = (self.xMax-self.xMin)/(self.N-1)
        self.result = sum( self.f(self.xMin+i*deltaX) for i in range(self.N-1))*deltaX
        
    def show(self):
        print (self.result)

        

examp = Integrator(1,3,200)
examp.integrate()
examp.show()