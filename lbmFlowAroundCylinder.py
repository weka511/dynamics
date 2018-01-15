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

import matplotlib.pyplot as plt,numpy as np
from matplotlib import cm

###### Flow definition ######################## #################################
maxIter   = 200000                # Total number of time iterations.
Re        = 38.125                # Reynolds number.
nx, ny    = 420, 180              # Number of lattice nodes.
ly        = ny-1                  # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9   # Coordinates of the cylinder.
uLB       = 0.04                  # Velocity in lattice units.
nulb      = uLB*r/Re;             # Viscoscity in lattice units.
omega     = 1 / (3*nulb+0.5)      # Relaxation parameter.

# In lecture, fin and fout are indexed as follows
# fin:   258       fout: 630
#        147             741
#        036             852

###### Lattice Constants #######################################################
v = np.array([ # weights for velocity calculation - first index matches fin
    [ 1,  1],  # Lower Left
    [ 1,  0],  # Left middle
    [ 1, -1],  # Upper left
    [ 0,  1],  # Upper centre
    [ 0,  0],  # Middle centre (stationary)
    [ 0, -1],  # Lower centre
    [-1,  1],  # Upper right
    [-1,  0],  # Middle right
    [-1, -1]   # Lower right
    ])

t = np.array([ # weights used in equilibrium calculation - allow dor differences in velocities
    1/36,   # Lower Left - [1,1]
    1/9,    # Left middle = [1,0]
    1/36,
    1/9,
    4/9,    # Middle (stationary)
    1/9,
    1/36, 
    1/9, 
    1/36
])

col1 = np.array([0, 1, 2])
col2 = np.array([3, 4, 5])
col3 = np.array([6, 7, 8])


def macroscopic(fin):
    '''
    Calculate density and velocity
    '''
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0,:,:] += v[i,0] * fin[i,:,:]
        u[1,:,:] += v[i,1] * fin[i,:,:]
    u /= rho
    return rho, u

def equilibrium(rho, u):
    '''
    Equilibrium distribution function (series expansion of Bolzmann)
    '''
    usqr = 3/2 * (u[0]**2 + u[1]**2)
    feq = np.zeros((9,nx,ny))
    for i in range(9):
        cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i] * (1 + cu + 0.5*cu**2 - usqr)
    return feq

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########

def obstacle_fun(x, y):
    '''
    Create a mask with 1/0 values, defining the shape of the obstacle.
    '''
    return (x-cx)**2+(y-cy)**2<r**2

obstacle = np.fromfunction(obstacle_fun, (nx,ny))

def inivel(d, x, y,perturbation=1e-4):
    '''
    Initial velocity profile: almost zero, with a slight perturbation to trigger
    the instability.
    '''
    return (1-d) * uLB * (1 + perturbation*np.sin(y/ly*2*np.pi))

vel = np.fromfunction(inivel, (2,nx,ny))

# Initialization of the populations at equilibrium with the given velocity.
fin = equilibrium(1, vel)

def visualize_velocity(time,u,Re,images = './images/',freq=100):
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
        plt.imshow(np.sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Reds)
        plt.title('Re={0}'.format(Re))
        plt.savefig('{0}vel.{1:04d}.png'.format(images,time//100))
        
if __name__=='__main__':
    import time
    start_time = time.time() 
    print ('Reynolds number={0}'.format(Re))
    for T in range(maxIter):
        fin[col3,-1,:] = fin[col3,-2,:]  # Right wall: outflow condition.
    
        rho, u = macroscopic(fin) # Compute macroscopic variables, density and velocity.
    
        # Left wall: inflow condition.
        u[:,0,:] = vel[:,0,:]
        rho[0,:] = 1/(1-u[0,0,:]) * ( np.sum(fin[col2,0,:], axis=0) +
                                      2*np.sum(fin[col3,0,:], axis=0) )
        # Compute equilibrium.
        feq = equilibrium(rho, u)
        fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:] #???
    
        fout = fin - omega * (fin - feq)  # Collision step.
    
        # Bounce-back condition for obstacle.
        for i in range(9):
            fout[i, obstacle] = fin[8-i, obstacle]
    
        # Streaming step.
        for i in range(9):
            fin[i,:,:] = np.roll(np.roll(fout[i,:,:], 
                                         v[i,0], 
                                         axis=0),
                                 v[i,1],
                                 axis=1)
            
        visualize_velocity(T,u,Re)
        
    print("--- Execution time for {0} steps = {1} seconds ---".format( maxIter, int(time.time() - start_time)))
    