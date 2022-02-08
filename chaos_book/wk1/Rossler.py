# Q1.4  Integrating Roessler system
from numpy             import array, linspace
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show
from RungeKutta        import RK4



#Parameters:
A = 0.2
B = 0.2
C = 5.7


def Velocity(ssp, t,
             a = A,
             b = B,
             c = C):
    """
    Velocity function for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
    """

    x, y, z = ssp  # Read state space points
    # Rossler flow equations:
    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    # Collect Rossler flow equations in a single NumPy array:
    vel = array([dxdt, dydt, dzdt], float)  # Velocity vector
    return vel


def Flow(ssp0, deltat):
    """
    Lagrangian description of the flow:
    This function integrates Rossler equation starting at ssp0 for deltat, and
    returns the final state space point.
    Inputs:
    ssp0: Initial state space point
    deltat: Integration time
    Outputs:
    sspdeltat: Final state space point
    """
    #Following numerical integration will return a 2 by 3(=d) solution array
    #where first row contains initial point ssp0, and the last row contains
    #final point
    sspSolution = odeint(Velocity, ssp0, [0.0, deltat])
    sspdeltat = sspSolution[-1, :]  # Read the final point to sspdeltat
    return sspdeltat


def StabilityMatrix(ssp):
    """
    Stability matrix for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp = [x, y, z]
    Outputs:
    A: Stability matrix evaluated at ssp. dxd NumPy array
       A[i, j] = del Velocity[i] / del ssp[j]
    """

    x, y, z = ssp  # Read state space points

    A = np.array([[0, -1, -1],
                  [None, None, None],
                  [None, None, None]], float)  # COMPLETE THIS LINE
    return A


def JacobianVelocity(sspJacobian, t):
    """
    Velocity function for the Jacobian integration

    Inputs:
    sspJacobian: (d+d^2)x1 dimensional state space vector including both the
                 state space itself and the tangent space
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    velJ = (d+d^2)x1 dimensional velocity vector
    """

    ssp = sspJacobian[0:3]  # First three elements form the original state
                            # space vector
    J = sspJacobian[3:].reshape((3, 3))  # Last nine elements corresponds to
                                         # the elements of Jacobian.
    #We used numpy.reshape function to reshape d^2 dimensional vector which
    #hold the elements of Jacobian into a dxd matrix.
    #See
    #http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    #for the reference for numpy.reshape function

    velJ = np.zeros(np.size(sspJacobian))  # Initiate the velocity vector as a
                                           # vector of same size with
                                           # sspJacobian
    velJ[0:3] = Velocity(ssp, t)
    #Last dxd elements of the velJ are determined by the action of
    #stability matrix on the current value of the Jacobian:
    velTangent = np.dot(StabilityMatrix(ssp), J)  # Velocity matrix for
                                                  #  the tangent space
    velJ[3:] = np.reshape(velTangent, 9)  # Another use of numpy.reshape, here
                                          # to convert from dxd to d^2
    return velJ


def Jacobian(ssp, t):
    """
    Jacobian function for the trajectory started on ssp, evolved for time t

    Inputs:
    ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
    t: Integration time
    Outputs:
    J: Jacobian of trajectory f^t(ssp). dxd NumPy array
    """
    #CONSTRUCT THIS FUNCTION
    #Hint: See the Jacobian calculation in CycleStability.py
    J = None

    return J

if __name__ == "__main__":
    tInitial = 0  # Initial time
    tFinal   = 2.0 # 5.881088455554846384  # Final time
    Nt       = 10000  # Number of time points to be used in the integration

    tArray   = linspace(tInitial, tFinal, Nt)  # Time array for solution
    ssp0     = array([9.269083709793489945,
                      0.0,
                      2.581592405683282632], float)  # Initial condition for the solution

    sspSolution = odeint(Velocity, ssp0, tArray)

    xt = sspSolution[:, 0]
    yt = sspSolution[:, 1]
    zt = sspSolution[:, 2]

    print((xt[-1], yt[-1], zt[-1]))  # Print final point



    fig = figure()  # Create a figure instance
    ax  = fig.gca(projection='3d')  # Get current axes in 3D projection
    ax.plot(xt, yt, zt)  # Plot the solution
    ax.set_xlabel('x')  # Set x label
    ax.set_ylabel('y')  # Set y label
    ax.set_zlabel('z')  # Set z label
    show()  # Show the figure
