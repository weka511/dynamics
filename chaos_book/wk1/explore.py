#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint
from matplotlib.pyplot import figure, show
from Rossler import    Velocity

if __name__ == "__main__":
    tInitial = 0  # Initial time
    tFinal   = 5.881088455554846384  # Final time
    Nt       = 10000  # Number of time points to be used in the integration

    tArray   = np.linspace(tInitial, tFinal, Nt)  # Time array for solution
    ssp0     = np.array([9.269083709793489945,
                      0.0,
                      2.581592405683282632], float)  # Initial condition for the solution

    sspSolution = odeint(Velocity, ssp0, tArray)

    xt = sspSolution[:, 0]
    yt = sspSolution[:, 1]
    zt = sspSolution[:, 2]

    print((xt[-1], yt[-1], zt[-1]))  # Print final point



    fig = figure()  # Create a figure instance
    ax  = fig.add_subplot(1,1,1,projection='3d')  # Get current axes in 3D projection
    ax.plot(xt, yt, zt)  # Plot the solution
    ax.set_xlabel('x')  # Set x label
    ax.set_ylabel('y')  # Set y label
    ax.set_zlabel('z')  # Set z label
    show()  # Show the figure
