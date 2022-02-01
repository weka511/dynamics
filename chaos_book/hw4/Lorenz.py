############################################################
# This file contains related functions for integrating and reducing
# Lorenz system.
# 
# please fill out C2, velocity(), stabilityMatrix(),
# integrator_with_jacob(), reduceSymmetry(), case3 and case4.
############################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.random import rand
from scipy.integrate import odeint

# G_ means global
G_sigma = 10.0
G_rho = 28.0
G_b = 8.0/3.0

# complete the definition of C^{1/2} operation matrix for Lorenz
# system. 
C2 = np.array([
        [None, None, None],
        [None, None, None],
        [None, None, None]
        ])


def velocity(stateVec, t):
    """
    return the velocity field of Lorentz system.
    stateVec : the state vector in the full space. [x, y, z]
    t : time is used since odeint() requires it. 
    """
    
    x = stateVec[0]
    y = stateVec[1]
    z = stateVec[2]
    
    # complete the flowing 3 lines.
    vx =  
    vy = 
    vz = 

    return np.array([vx, vy, vz])

def stabilityMatrix(stateVec):
    """
    return the stability matrix at a state point.
    stateVec: the state vector in the full space. [x, y, z]
    """
    
    x = stateVec[0]; y = stateVec[1]; z = stateVec[2];
    # fill out the following matrix.
    stab = np.array([
            [None, None, None],
            [None, None, None],
            [None, None, None]
            ])
    
    return stab

def integrator(init_x, dt, nstp):
    """
    The integator of the Lorentz system.
    init_x: the intial condition
    dt : time step
    nstp: number of integration steps.
    
    return : a [ nstp x 3 ] vector 
    """

    state = odeint(velocity, init_x, np.arange(0, dt*nstp, dt))
    return state

def integrator_with_jacob(init_x, dt, nstp):
    """
    integrate the orbit and the Jacobian as well. The meaning 
    of input parameters are the same as 'integrator()'.
    
    return : 
            state: a [ nstp x 3 ] state vector 
            Jacob: [ 3 x 3 ] Jacobian matrix
    """

    # Please fill out the implementation of this function.
    # You can go back to the previous homework to see how to
    # integrate state and Jacobian at the same time.
    
    
    state = 
    Jacob = 
    
    return state, Jacob

def reduceSymmetry(states):
    """
    reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
    (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)
    
    states: trajectory in the full state space. dimension [nstp x 3]
    return: states in the invariant polynomial basis. dimension [nstp x 3]
    """
    
    m, n = states.shape

    # please fill out the transformation from states to reducedStates.
    
    reducedStates = 
    
    return reducedStates

def plotFig(orbit):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
    plt.show()


if __name__ == "__main__":
    
    case = 1
   
    # case 1: try a random initial condition
    if case == 1:
        x0 = rand(3)
        dt = 0.005
        nstp = 50.0/dt
        orbit = integrator(x0, dt, nstp)
        reduced_orbit = reduceSymmetry(orbit)
    
        plotFig(orbit)
        plotFig(reduced_orbit)

    # case 2: periodic orbit
    if case == 2:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149
        nstp = 156
        orbit_doulbe = integrator(x0, dt, nstp*2)
        orbit = orbit_doulbe[:nstp, :] # one prime period
        reduced_orbit = reduceSymmetry(orbit)

        plotFig(orbit_doulbe)
        plotFig(reduced_orbit)

    # case 3 : calculate Floquet multipliers and Floquet vectors associated
    # with the full periodic orbit in the full state space.
    # Please check that one of Floquet vectors is in the same/opposite
    # direction with velocity field at x0.
    if case == 3:    
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149 # integration time step
        nstp = 156 # number of integration steps => T = nstp * dt
        
        # please fill out the part to calculate Floquet multipliers and
        # vectors.



    # case 4: calculate Floquet multipliers and Flqouet vectors associated
    # with the prime period. 
    if case == 4:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149
        nstp = 156
        
        # please fill out the part to calculate Floquet multipliers and
        # vectors.
    
    
