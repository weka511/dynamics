#!/usr/bin/python3
# Copyright (C) 2015 Universite de Geneve, Switzerland
# E-mail contact: jonas.latt@unige.ch
#
# 2D flow around a cylinder
#

# Use the Python code provided in the course,
# to approximately determine the critical Reynold number,
# i.e. the lowest Reynold number for which the flow around the cylinder
# enters an unsteady regime after a sufficient number of iterations.


# For the purpose of this project, it is sufficient to find the critical 
# Reynolds number within an accuracy of +/-5. You will need to run the Python
# code several times at different Reynolds numbers, but in each run,
# you are not required to run the code for more than 200 000 time steps.

# You may want to consult the following Wikipedia article to make
# sure you understand the difference between a steady and an unsteady
# flow: https://en.wikipedia.org/wiki/Fluid_dynamics#Steady_vs_unsteady_flow

# For this exercise, we consider a flow configuration to be steady 
# if the velocity norm ||u|| is time independent after a sufficient number of iterations.
# Otherwise, we assume that it remains unsteady forever.
# Note that the cylinder is spatially immobile and that the frame of reference is stationary with respect to the cylinder.

# It is also interesting to point out that the critical Reynolds number
# depends on the flow configuration. In an exterior flow, in which the flow is not, 
# as in our example, restricted by lateral walls, the critical Reynolds number would be larger.

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm

###### Flow definition #########################################################
maxIter = 200000 # Total number of time iterations.
Re = 10.0         # Reynolds number.
nx, ny = 420, 180 # Numer of lattice nodes.
ly = ny-1         # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*r/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.

###### Lattice Constants #######################################################
v = array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
            [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ])
t = array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])

col1 = array([0, 1, 2])
col2 = array([3, 4, 5])
col3 = array([6, 7, 8])



def macroscopic(fin):
    rho = sum(fin, axis=0)
    u = zeros((2, nx, ny))
    for i in range(9):
        u[0,:,:] += v[i,0] * fin[i,:,:]
        u[1,:,:] += v[i,1] * fin[i,:,:]
    u /= rho
    return rho, u

def equilibrium(rho, u):
    '''
    Equilibrium distribution function.
    '''
    usqr = 3/2 * (u[0]**2 + u[1]**2)
    feq = zeros((9,nx,ny))
    for i in range(9):
        cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i] * (1 + cu + 0.5*cu**2 - usqr)
    return feq

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
# Creation of a mask with 1/0 values, defining the shape of the obstacle.
def obstacle_fun(x, y):
    return (x-cx)**2+(y-cy)**2<r**2

obstacle = fromfunction(obstacle_fun, (nx,ny))

# Initial velocity profile: almost zero, with a slight perturbation to trigger
# the instability.
def inivel(d, x, y):
    return (1-d) * uLB * (1 + 1e-4*sin(y/ly*2*pi))

vel = fromfunction(inivel, (2,nx,ny))

# Initialization of the populations at equilibrium with the given velocity.
fin = equilibrium(1, vel)

def visualize_velocity(time,u,images = './images/',freq=100):
    '''
    Visualize the velocity.
    
        Parameters:
            time
            u
            images
            freq
    '''
    if (time%freq==0):
        plt.clf()
        plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Reds)
        plt.savefig('{0}vel.{1:04d}.png'.format(images,time//100))
        
if __name__=='__main__':
    import time
    start_time = time.time()    
    for T in range(maxIter):
        fin[col3,-1,:] = fin[col3,-2,:]  # Right wall: outflow condition.
    
        rho, u = macroscopic(fin) # Compute macroscopic variables, density and velocity.
    
        # Left wall: inflow condition.
        u[:,0,:] = vel[:,0,:]
        rho[0,:] = 1/(1-u[0,0,:]) * ( sum(fin[col2,0,:], axis=0) +
                                      2*sum(fin[col3,0,:], axis=0) )
        # Compute equilibrium.
        feq = equilibrium(rho, u)
        fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]
    
        fout = fin - omega * (fin - feq)  # Collision step.
    
        # Bounce-back condition for obstacle.
        for i in range(9):
            fout[i, obstacle] = fin[8-i, obstacle]
    
        # Streaming step.
        for i in range(9):
            fin[i,:,:] = roll(roll(fout[i,:,:], 
                                   v[i,0], 
                                   axis=0),
                              v[i,1],
                              axis=1)
            
        visualize_velocity(T,u)
        
    print("--- Execution time for {0} steps = {1} seconds ---".format( maxIter, int(time.time() - start_time)))

