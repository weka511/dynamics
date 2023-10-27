#!/usr/bin/env python

# Copyright (C) 2015-2023 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>

import math

class Driver(object):
    def __init__(self,integrator,h_minimum,h,h_maximum,epsilon,mult=0.01):
        self.integrator = integrator
        self.h_minimum = h_minimum
        self.h_maximum = h_maximum
        self.epsilon = epsilon
        self.h = h
        self.min_epsilon = mult*epsilon

    def step(self,y):
        try:
            y1 = self.integrator.step(self.h,y)
            y11 = self.integrator.step(0.5*self.h,self.integrator.step(0.5*self.h,y))
            error = self.integrator.distance(y1,y11)
            if error > self.epsilon:
                self.h *= (self.epsilon/error) ** (1.0/self.integrator.order)
                return self.step(y)
            if error < self.min_epsilon:
                if error > 0:
                    self.h *= (self.min_epsilon/error)**(1.0/self.integrator.order)
                else:
                    self.h *= 2.0
                if self.h > self.h_maximum:
                    self.h = self.h_maximum

            return y11
        except ImplicitRungeKutta.Failed:
            self.h *= 0.5
            return self.step(y)

# see https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Gauss.E2.80.93Legendre_methods

class ImplicitRungeKutta(object):
    class Failed(Exception):
        def __init__(self, value):
                self.value = value
        def __str__(self):
            return repr(self.value)

    def __init__(self, dy,max_iterations,iteration_error,order):
        self.dy = dy
        self.max_iterations = max_iterations
        self.iteration_error = iteration_error
        self.order = order

    def distance(self,k,k_new):
            return max([abs(a-b) for (a,b) in zip(k,k_new)])


    def step(self,h,y):
        k=[[0 for col in y] for row in range(len(self.b))]
        for i in range(self.max_iterations):
            k_new = self.iterate(h,y,k)
            if min([self.distance(k0,k1) for (k0,k1) in zip(k,k_new)]) < self.iteration_error:
                yy = [y0 for y0 in y]
                for l in range(len(yy)):
                    inner_product = 0
                    for j in range(self.s):
                        inner_product += self.b[j]*k_new[j][l]
                    yy[l] += h*inner_product
                return yy
            else:
                k = k_new
        self.fail()

    # assume that each row of the matrix is a single vector a = [[row1,...]]
    def iterate(self,h,y,k):
        result=[[0 for col in y] for row in range(self.s)]
        for i in range(self.s):
            yy = [y0 for y0 in y]
            for l in range(len(k[i])):
                inner_product = 0
                for j in range(self.s):
                    inner_product += self.a[i][j]*k[j][l]
                yy[l] += h*inner_product
            result[i] = self.dy(yy)
        return result

    def fail(self):
        raise ImplicitRungeKutta.Failed(                     \
            'Failed to Converge within {0} after {1} iterations'.format(  \
                self.iteration_error,                          \
                self.max_iterations))

class ImplicitRungeKutta2(ImplicitRungeKutta):
    def __init__(self,dy,max_iterations,iteration_error):
        super(ImplicitRungeKutta2, self).__init__(dy,max_iterations,iteration_error,4)
        r3=math.sqrt(3.0)
        self.a=[
            [0.25,        0.25-r3/6.0],
            [0.25+r3/6.0, 0.25],
        ]
        self.b=[
            0.5, 0.5]
        self.c=[
            0.5-r3/6,
            0.5+r3/6
        ]
        self.s=len(self.b)

class ImplicitRungeKutta4(ImplicitRungeKutta):
    def __init__(self,dy,max_iterations,iteration_error):
        super(ImplicitRungeKutta4, self).__init__(dy,max_iterations,iteration_error,6)
        r15=math.sqrt(15.0)
        self.a=[
            [5.0/36.0,          2.0/9.0-r15/15.0, 5.0/36.0-r15/30.0],
            [5.0/36.0+r15/24.0, 2.0/9.0,          5.0/36.0-r15/24.0],
            [5.0/36.0+r15/30.0, 2.0/9.0+r15/15.0, 5.0/36.0]
        ]
        self.b=[
            5.0/18.0,
            4.0/9.0,
            5.0/18.0]
        self.c=[
            0.5-r15/10.0,
            0.5,
            0.5+r15/10.0
        ]
        self.s=len(self.b)

if __name__=='__main__':
    rk=ImplicitRungeKutta2(lambda y: [y[1],-y[0]],10,0.000000001)
    driver = Driver(rk,0.000000001,0.5,1.0,0.000000001)
    import matplotlib.pyplot as plt
    try:
        nn=1000
        y=[1,0]
        xs=[]
        ys=[]
        for i in range(nn):
            y= driver.step(y)
            xs.append(y[0])
            ys.append(y[1])
        plt.plot(xs,ys)
        plt.show()
    except ImplicitRungeKutta.Failed as e:
        print ("caught!",e)



