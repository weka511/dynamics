'''Rossler Flow Example from Week 2'''

from matplotlib.pyplot import figure, show
from numpy             import array, dot, linspace, reshape, size, zeros
from scipy.integrate   import odeint

a = 0.2
b = 0.2
c = 5.7

def Velocity(ssp, t):
    '''
    Velocity function for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
    '''

    x, y, z = ssp  # Read state space points
    # Rossler flow equations:
    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    # Collect Rossler flow equations in a single NumPy array:
    vel = array([dxdt, dydt, dzdt], float)  # Velocity vector
    return vel


def Flow(ssp0, deltat):
    '''
    Lagrangian description of the flow:
    This function integrates Rossler equation starting at ssp0 for deltat, and
    returns the final state space point.
    Inputs:
    ssp0: Initial state space point
    deltat: Integration time
    Outputs:
    sspdeltat: Final state space point
    '''
    #Following numerical integration will return a 2 by 3(=d) solution array
    #where first row contains initial point ssp0, and the last row contains
    #final point
    sspSolution = odeint(Velocity, ssp0, [0.0, deltat])
    sspdeltat = sspSolution[-1, :]  # Read the final point to sspdeltat
    return sspdeltat


def StabilityMatrix(ssp):
    '''
    Stability matrix for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp = [x, y, z]
    Outputs:
    A: Stability matrix evaluated at ssp. dxd NumPy array
       A[i, j] = del Velocity[i] / del ssp[j]
    '''

    x, y, z = ssp  # Read state space points

    return array([[0, -1, -1],
                  [1, a, 0],
                  [z, 0, x-c]],
                 float)



def JacobianVelocity(sspJacobian, t):
    '''
    Velocity function for the Jacobian integration

    Inputs:
    sspJacobian: (d+d^2)x1 dimensional state space vector including both the
                 state space itself and the tangent space
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    velJ = (d+d^2)x1 dimensional velocity vector
    '''

    ssp = sspJacobian[0:3]  # First three elements form the original state
                            # space vector
    J = sspJacobian[3:].reshape((3, 3))  # Last nine elements corresponds to
                                         # the elements of Jacobian.
    #We used numpy.reshape function to reshape d^2 dimensional vector which
    #hold the elements of Jacobian into a dxd matrix.
    #See
    #http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    #for the reference for numpy.reshape function

    velJ = zeros(size(sspJacobian))  # Initiate the velocity vector as a
                                           # vector of same size with
                                           # sspJacobian
    velJ[0:3] = Velocity(ssp, t)
    #Last dxd elements of the velJ are determined by the action of
    #stability matrix on the current value of the Jacobian:
    velTangent = dot(StabilityMatrix(ssp), J)  # Velocity matrix for
                                                  #  the tangent space
    velJ[3:] = reshape(velTangent, 9)  # Another use of numpy.reshape, here
                                          # to convert from dxd to d^2
    return velJ


def Jacobian(ssp, t):
    '''
    Jacobian function for the trajectory started on ssp, evolved for time t

    Inputs:
    ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
    t: Integration time
    Outputs:
    J: Jacobian of trajectory f^t(ssp). dxd NumPy array
    '''
    #CONSTRUCT THIS FUNCTION
    #Hint: See the Jacobian calculation in CycleStability.py
    J = None

    return J

if __name__ == '__main__':
    tInitial = 0
    tFinal   = 100
    Nt       = 10000

    tArray = linspace(tInitial, tFinal, Nt)
    ssp0 = array([1.0,
                  1.0,
                  1.0],
                 float)  # Initial condition for the solution

    sspSolution = odeint(Velocity, ssp0, tArray)

    xt = sspSolution[:, 0]
    yt = sspSolution[:, 1]
    zt = sspSolution[:, 2]

    print((xt[-1], yt[-1], zt[-1]))  # Print final point

    fig = figure()
    ax  = fig.add_subplot(projection='3d')
    ax.plot(xt, yt, zt)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()
