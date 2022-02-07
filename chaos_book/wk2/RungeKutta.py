import numpy as np  # Import numpy


def RK4(velocityFunction, initialCondition, timeArray):
    """
    Runge-Kutta 4 Integrator.
    Inputs:
    VelocityFunction: Function name to integrate
                      this function must have two inputs namely state space
                      vector and time. For example: velocity(ssp, t)
    InitialCondition: Initial condition, 1xd NumPy array, where d is the
                      dimension of the state space
    TimeArray: 1 x Nt NumPy array which contains instances for the solution
               to be returned.
    Outputs:
    SolutionArray: d x Nt NumPy array which contains numerical solution of the
                   ODE.
    """
    #Generate the solution array to fill in:
    SolutionArray = np.zeros((np.size(timeArray, 0),
                              np.size(initialCondition, 0)))
    #Assign the initial condition to the first element:
    SolutionArray[0, :] = initialCondition

    for i in range(0, np.size(timeArray) - 1):
        #Read time element:
        deltat = timeArray[i + 1] - timeArray[i]
        #Runge Kutta k's:
        k1 = deltat * velocityFunction(SolutionArray[i], timeArray[i])
        k2 = None  # COMPLETE THIS LINE
        k3 = None  # COMPLETE THIS LINE
        k4 = None  # COMPLETE THIS LINE
        #Next integration step:
        SolutionArray[i + 1] = SolutionArray[i] + None  # COMPLETE THIS LINE
    return SolutionArray

if __name__ == "__main__":
    #This block will be evaluated if this script is called as the main routine
    #and will be ignored if this file is imported from another script.
    #
    #This is a handy structure in Python which lets us test the functions
    #in a package

    #In order to test our integration routine, we are going to define Harmonic
    #Oscillator equations in a 2D state space:
    def velocity(ssp, t):
        """
        State space velocity function for 1D Harmonic oscillator

        Inputs:
        ssp: State space vector
        ssp = (x, v)
        t: Time. It does not effect the function, but we have t as an imput so
           that our ODE would be compatible for use with generic integrators
           from scipy.integrate

        Outputs:
        vel: Time derivative of ssp.
        vel = ds sp/dt = (v, - (k/m) x)
        """
        #Parameters:
        k = 1.0
        m = 1.0
        #Read inputs:
        x, v = ssp  # Read x and v from ssp vector
        #Construct velocity vector and return it:
        vel = np.array([v, - (k / m) * x], float)
        return vel

    #Generate an array of time points for which we will compute the solution:
    tInitial = 0
    tFinal = 8
    Nt = 800  # Number of points time points in the interval tInitial, tFinal
    tArray = np.linspace(tInitial, tFinal, Nt)

    #Initial condition for the Harmonic oscillator:
    ssp0 = np.array([1.0, 0], float)

    #Compute the solution using Runge-Kutta routine:
    sspSolution = RK4(velocity, ssp0, tArray)

    #from scipy.integrate import odeint
    #sspSolution = odeint(velocity, ssp0, tArray)
    xSolution = sspSolution[:, 0]
    vSolution = sspSolution[:, 1]

    print(xSolution[-1])

    #Import functions which we need for plotting our results:
    from pylab import subplot, plot, xlabel, ylabel, show

    subplot(2, 1, 1)
    plot(tArray, xSolution)
    ylabel('x(t)')

    subplot(2, 1, 2)
    plot(tArray, vSolution)
    xlabel('t (s)')
    ylabel('v(t)')

    show()
