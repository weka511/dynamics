############################################################
# This file contains related functions for integrating and reducing
# Lorenz system.
#
# please fill out C2, velocity(), stabilityMatrix(),
# integrator_with_jacob(), reduceSymmetry(), case3 and case4.
############################################################

from numpy             import arange, array, zeros_like
from matplotlib.pyplot import figure, show, suptitle
from numpy.random      import rand
from scipy.integrate   import odeint
from argparse          import ArgumentParser

sigma = 10.0
rho   = 28.0
b     = 8.0/3.0

# complete the definition of C^{1/2} operation matrix for Lorenz
# system.
C2 = array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
        ])


def velocity(stateVec, t):
    '''
    return the velocity field of Lorentz system.
    stateVec : the state vector in the full space. [x, y, z]
    t : time is used since odeint() requires it.
    '''

    x = stateVec[0]
    y = stateVec[1]
    z = stateVec[2]

    vx = sigma * (y-x)
    vy = rho * x -x -x * z
    vz = x*y - b*z

    return array([vx, vy, vz])

def stabilityMatrix(stateVec):
    '''
    return the stability matrix at a state point.
    stateVec: the state vector in the full space. [x, y, z]
    '''

    x = stateVec[0]; y = stateVec[1]; z = stateVec[2];
    # fill out the following matrix.
    stab = array([
            [-sigma, sigma, 0],
            [rho-z,  -1 , -x],
            [y,      x,   -b]
            ])

    return stab

def integrator(init_x, dt, nstp):
    '''
    The integator of the Lorentz system.
    init_x: the intial condition
    dt : time step
    nstp: number of integration steps.

    return : a [ nstp x 3 ] vector
    '''

    state = odeint(velocity, init_x, arange(0, dt*nstp, dt))
    return state

def integrator_with_jacob(init_x, dt, nstp):
    '''
    integrate the orbit and the Jacobian as well. The meaning
    of input parameters are the same as 'integrator()'.

    return :
            state: a [ nstp x 3 ] state vector
            Jacob: [ 3 x 3 ] Jacobian matrix
    '''

    # Please fill out the implementation of this function.
    # You can go back to the previous homework to see how to
    # integrate state and Jacobian at the same time.


    state = None
    Jacob = None

    return state, Jacob

def reduceSymmetry(states):
    '''
    reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
    (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)

    states: trajectory in the full state space. dimension [nstp x 3]
    return: states in the invariant polynomial basis. dimension [nstp x 3]
    '''

    x,y,z              = states[:,0], states[:,1], states[:,2]
    (u, v)             = (x*2 - y*2, 2*x*y)
    reducedStates      = zeros_like(states)
    reducedStates[:,0] = u
    reducedStates[:,1] = v
    reducedStates[:,2] = z
    return reducedStates

def plotFig(orbit):
    fig = figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
    show()

def plot_orbits(orbit,reduced_orbit,case=None):
    fig = figure(figsize=(8,6))
    suptitle(f'Case {case}')
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
    ax.set_title('Orbit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title('Reduced')
    ax.plot(reduced_orbit[:,0], reduced_orbit[:,1], reduced_orbit[:,2])
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('z')
    show()

if __name__ == '__main__':
    parser = ArgumentParser('Q4.3 Symmetry of Lorenz Flow')
    parser.add_argument('case',
                        type    = int,
                        choices = [1,2,3,4])
    args = parser.parse_args()

    # case 1: try a random initial condition
    if args.case == 1:
        x0            = rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        orbit         = integrator(x0, dt, nstp)
        reduced_orbit = reduceSymmetry(orbit)
        plot_orbits(orbit,reduced_orbit,case=args.case)


    # case 2: periodic orbit
    if args.case == 2:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149
        nstp = 156
        orbit_double = integrator(x0, dt, nstp*2)
        orbit = orbit_double[:nstp, :] # one prime period
        reduced_orbit = reduceSymmetry(orbit)

        plotFig(orbit_double)
        plotFig(reduced_orbit)

    # case 3 : calculate Floquet multipliers and Floquet vectors associated
    # with the full periodic orbit in the full state space.
    # Please check that one of Floquet vectors is in the same/opposite
    # direction with velocity field at x0.
    if args.case == 3:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149 # integration time step
        nstp = 156 # number of integration steps => T = nstp * dt

        # please fill out the part to calculate Floquet multipliers and
        # vectors.



    # case 4: calculate Floquet multipliers and Flqouet vectors associated
    # with the prime period.
    if args.case == 4:
        x0 = np.array([ -0.78844208,  -1.84888176,  18.75036186])
        dt = 0.0050279107820829149
        nstp = 156

        # please fill out the part to calculate Floquet multipliers and
        # vectors.


