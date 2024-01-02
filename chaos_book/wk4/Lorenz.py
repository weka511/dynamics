#!/usr/bin/env python

'''
    Q4.3 Symmetry of Lorenz Flow
'''
from argparse import ArgumentParser
import numpy as np
from matplotlib.pyplot import figure, show
from scipy.integrate import solve_ivp
from scipy.linalg import eig, norm

sigma = 10.0
rho   = 28.0
b     = 8.0/3.0



def Velocity(t, stateVec):
    '''
    Calculate the velocity field of Lorentz system.

    Parameters:
        stateVec : the state vector in the full space. [x, y, z]
        t : time is unused, but solve_ivp(...) requires it.

    Returns:
       velocity field
    '''

    x = stateVec[0]
    y = stateVec[1]
    z = stateVec[2]

    return np.array([sigma * (y-x),
                     rho*x - y - x*z,
                     x*y - b*z])

def stabilityMatrix(stateVec):
    '''
    Calculate the stability matrix at a state point.

    Parameters:
        stateVec the state vector in the full space. [x, y, z]
    '''

    x = stateVec[0]
    y = stateVec[1]
    z = stateVec[2]

    return np.array([
        [-sigma, sigma, 0],
        [rho-z,  -1, -x],
        [y,      x,   -b]
    ])


def integrator(init_x, dt, nstp):
    '''
    Integate the Lorentz system.
    Parameters:
        init_x: the intial condition
        dt : time step
        nstp: number of integration steps.

    Returns:
        a [ 3 x nstp ] vector
    '''
    return solve_ivp(Velocity, (0, dt*nstp),init_x, t_eval=np.linspace(0, dt*nstp, nstp)).y


def integrator_with_jacob(init_x, dt, nstp):
    '''
    integrate the orbit and the Jacobian as well. The meaning
    of input parameters are the same as 'integrator()'.

    return :
            state: a [ nstp x 3 ] state vector
            Jacob: [ 3 x 3 ] Jacobian matrix
    '''
    def JacobianVelocity(t, sspJacobian):
        ssp = sspJacobian[0:d]                 # First three elements form the original state space vector
        J = sspJacobian[d:].reshape((d, d))  # Last nine elements corresponds to the elements of Jacobian.
        velJ = np.zeros(np.size(sspJacobian))         # Initiate the velocity vector as a vector of same size as sspJacobian
        velJ[0:d] = Velocity(t, ssp)
        velTangent = np.dot(stabilityMatrix(ssp), J)     # Velocity matrix for  the tangent space
        velJ[d:] = np.reshape(velTangent, d*d)           # Last dxd elements of the velJ are determined by the action of
                                                      # stability matrix on the current value of the Jacobian:
        return velJ

    d = len(init_x)
    Jacobian0  = np.identity(d)
    sspJacobian0 = np.zeros(d + d ** 2)
    sspJacobian0[0:d] = init_x
    sspJacobian0[d:] = np.reshape(Jacobian0, d**2)
    sspJacobianSolution = solve_ivp(JacobianVelocity,
                                    (0, dt*nstp),
                                    sspJacobian0,
                                    t_eval=np.linspace(0, dt*nstp, nstp)).y
    state = sspJacobianSolution[0:d,-1]
    Jacob = sspJacobianSolution[d:,-1].reshape((d, d))

    return state, Jacob

def reduceSymmetry(states):
    '''
    Reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
    (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)

    Paremeters:
        states: trajectory in the full state space. dimension [3 x nstp]

    Returns:
        states in the invariant polynomial basis. dimension [3 x nstp]
    '''

    x,y,z = states[0,:], states[1,:], states[2:,]
    (u, v) = (x**2 - y**2, 2*x*y)
    reducedStates = np.empty_like(states)
    reducedStates[0,:] = u
    reducedStates[1,:] = v
    reducedStates[2,:] = z
    return reducedStates


def plot_orbits(orbit,reduced_orbit,case=None):
    '''
    Plot both the orbit and reduced orbit in two separate figures

    Parameters:
        orbit
        reduced_orbit
        case
    '''
    fig = figure(figsize=(20,20))
    fig.suptitle(f'Case {case}')
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(orbit[0,:], orbit[1,:], orbit[2,:], markersize = 1)
    ax1.set_title('Orbit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Reduced')
    ax2.plot(reduced_orbit[0,:], reduced_orbit[1,:], reduced_orbit[2,:], markersize = 1)
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.set_zlabel('z')
    show()

if __name__ == '__main__':
    parser = ArgumentParser(__doc__)
    parser.add_argument('case',
                        type    = int,
                        choices = [1,2,3,4])
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    match(args.case):
        case 1: # case 1: try a random initial condition
            x0 = rng.uniform(0,1,size=3)
            dt = 0.005
            nstp = int(50.0/dt)
            orbit = integrator(x0, dt, nstp)
            reduced_orbit = reduceSymmetry(orbit)
            plot_orbits(orbit,reduced_orbit,case=args.case)

        case 2: #periodic orbit
            x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
            dt = 0.0050279107820829149
            nstp = 156
            orbit_double = integrator(x0, dt, nstp*2)
            orbit = orbit_double[:,:nstp] # one prime period
            reduced_orbit = reduceSymmetry(orbit)
            plot_orbits(orbit,reduced_orbit,case=args.case)

        case 3:# calculate Floquet multipliers and Floquet vectors associated
               # with the full periodic orbit in the full state space.
            x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
            dt = 0.0050279107820829149 # integration time step
            nstp = 156 # number of integration steps => T = nstp * dt
            state, Jacob = integrator_with_jacob(x0, dt, 2*nstp)
            eigenValues, eigenVectors = eig(Jacob)
            vel  = Velocity(2*nstp,state)
            vel /= norm(vel)
            print (f'Eigenvalues: {eigenValues}')
            print (f'Gradient,{vel}')
            # Check that one of Floquet vectors is in the same/opposite
            # direction with velocity field at x0.
            for i in range(len(eigenValues)):
                print (eigenVectors[:,i], norm(eigenVectors[:,i]), np.dot(eigenVectors[:,i],vel))


        case 4:# calculate Floquet multipliers and Floquet vectors associated with the prime period.
            C = np.array([ #  C^{1/2} operation matrix for Lorenz system.
                [-1, 0, 0],
                [0,  -1, 0],
                [0,  0, 1]])
            x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
            dt = 0.0050279107820829149
            nstp = 156
            _, Jacob = integrator_with_jacob(x0, dt, nstp)
            eigenValues, eigenVectors = eig(np.dot(C,Jacob))
            print (eigenValues)
