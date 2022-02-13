############################################################
#
# In this template, Henon map is implemented in a class because
# we need to change the parameters for future homeworks and
# it is much more convenient to use class here than global
# parameters.
# If you are not familiar with class object in python,
# have a look at the tutorial here:
# https://docs.python.org/2/tutorial/classes.html
#
# please set case = 1 to validate your implementation first.
# Then go to case2, case3 and case4.
# 
############################################################
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import rand
from scipy.interpolate import splrep, splev
from scipy.optimize import fsolve

class Henon:
    """
    Class Henon contains functions for both forward and 
    backward Henon map iteration.
    """
    def __init__(self, a=6, b=-1):
        """
        initialization function which will be called every time you
        create an object instance. In this case, it initializes 
        parameter a and b, which are the parameters of Henon map.
        """
        self.a = a
        self.b = b
        
    def oneIter(self, stateVec):
        """
        forward iterate for one step.

        stateVec: the current state. dimension : [1 x 2] numpy.array  
        return: the next state. dimension : [1 x 2]
        """
        x = stateVec[0]; 
        y = stateVec[1];

        stateNext =  

        return stateNext
    
    def multiIter(self, stateVec, NumOfIter):
        """
        forward iterate for multiple steps.
        stateVec: the current state. dimension : [1 x 2] numpy.array
        NumOfIter: number of iterations
        
        return: the current state and the furture 'NumOfIter' states.
                dimension [NumOfIter+1 x 2]
        Hint : numpy.vstack()
        """
        
        state = 

        return state

    def Jacob(self, stateVec):
        """
        The Jacobian for forward map at state point 'stateVec'.
        
        stateVec: the current state. dimension : [1 x 2] numpy.array
        """
        x = stateVec[0]; 
        y = stateVec[1];

        jacobian = 
        
        return jacobian 
            
    def oneBackIter(self, stateVec):
        """
        backward iterate for one step.

        stateVec: the current state. dimension : [1 x 2] numpy.array  
        return: the previous state. dimension : [1 x 2]
        """
        x = stateVec[0]; 
        y = stateVec[1];

        statePrev = 
        return statePrev

    def multiBackIter(self, stateVec, NumOfIter):
        """
        backward iterate for multiple steps.
        stateVec: the current state. dimension : [1 x 2] numpy.array
        NumOfIter: number of backward iterations
        
        return: the current state and the pervious 'NumOfIter' states.
                dimension [NumOfIter+1 x 2]
        """
        state = stateVec
        tmpState = stateVec
        for i in range(NumOfIter):
            tmp = self.oneBackIter(tmpState)
            tmpState = tmp
            state = np.vstack((state, tmpState))
            
        return state


if __name__ == '__main__':
    
    case = 4
    
    if case == 1:
        """
        Validate your implimentation of Henon map.
        Note here we use a=1.4, b=0.3 in this valication
        case. For other cases in this homework, we use a=6, b=-1.
        Actually, these two sets of parameters are both important 
        since we will come back to this model when discussing invariant measure.
        """
        henon = Henon(1.4, 0.3) # creake a Henon instance
        states = henon.multiIter(rand(2), 1000) # forward iterations
        states_bac = henon.multiBackIter(states[-1,:], 10) # backward iterations
        
        # check your implementation of forward map
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.scatter(states[:,0], states[:, 1], edgecolor='none') 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('(a)')
        plt.show()
        
        # check the correctness of backward map. The first 10 states_bac should
        # be the last 10 states in reverse order.
        # Note, backward map is very unstable for a=1.4, b=0.3,
        # so we only iterate backward for 10 times.
        print states[-10:, :]
        print '======'
        print states_bac[:10,:]
        print '======'

        # check the Jacobian matrix at (0.1, 0.2).
        # for a = 1.4, b = 0.3, the output should be
        # [[-0.28  0.3 ]
        #  [ 1.    0.  ]]

        print henon.Jacob(np.array([0.1, 0.2]))
        
        
    if case == 2:
        """
        Try to obtain the stable and unstable manifold for 
        equilibrium '0' in Henon map with parameter a=6, b=-1. 

        Plotting unstable/stable manifold is a difficult task in general. 
        We need to moniter a lot of variables, like the distances between points
        along the manifold, the angle formed by adjacent 3 points on the manifold, etc. 
        However, for Henon map, a simple algorithm is enough for demonstration purpose.
        The algorithm works as follows:
        
        Unstable manifold: start from a point close to equilibrium '0', in the direction of
        the expanding eigenvector: ' eq0 + r0 * V_e '. Here 'eq0' is the equilibrium,
        'r0' is a small number, 'V_e' is the expanding eigen direction of equilibrium '0'. 
        The image of
        this point after one forward iteration should be very close to 
        ' eq0 + r0 * \Lambda_e * V_e', where
        '\Lambda_e' is the expanding multiplier. We can confidently think that these two 
        points are sitting on the unstable manifold of 'eq0' since 'r0' is very small.
        Now, we interpolate linearly between these two points and get total 'N' points. 
        If we iterate these 'N' points forward for 'NumOfIter' times, we get the unstable
        manifold within some length.
        The crutial part of this method is that when these 'N' state points are being iterated,
        the spacings between them get larger and larger, so we need to use a relative large 
        value 'N' to ensure that these state points are not too far away from each other.
        The following formula is used to determine 'N' ( please figure out its meaning
        and convince yourself ):
        
        ( (\Lamba_e - 1) * r0 / N ) * (\Lambda_e)^NumOfIter = tol  (*)
        
        Here, 'tol' is the tolerance distance between adjacent points in the unstable manifold.

        Stable manifold: If we reverse the dynamics, then the stable direction becomes the 
        ustable direction, so we can use the same method as above to obtain stable manifold.

        """
        henon = Henon() # use the default parameters: a=6, b=-1
        # get the two equilbria of this map. equilibrium '0' should have smaller x coordinate.
        eq0 = np.array([, ]) # equilibrium '0' 
        eq1 = np.array([, ]) # equilibrium '1'

        # get the expanding multiplier and eigenvectors at equilibrium '0'
        Lamba_e = # expanding multiplier 
        Ev = # expanding eigenvector

        NumOfIter = 5 # number of iterations used to get stable/unstable manifold
        tol = 0.1 # tolerance distance between adjacent points in the manifold
        r0 = 0.0001 # small length 
        N = # implement the formula (*). Note 'N' should be an integer.
        delta_r = (Lamba_e-1)*r0 / N # initial spacing between points in the manifold

        # generate the unstable manifold. Note we do not use Henon.multiIter() here
        # since we want to keep the ordering of the points along the manifold.
        uManifold = eq0
        states = np.zeros([N,2])
        for i in range(N): # get the initial N points
            states[i,:] = eq0 + (r0 + delta_r*i)*Ev
        uManifold = np.vstack((uManifold, states))

        for i in range(NumOfIter):
            for j in range(N): # update these N points along the manifold
                states[j,:] = henon.oneIter(states[j,:]);
            uManifold = np.vstack((uManifold, states))


        # Please fill out this part to generate the stable manifold.
        # Check whether the stable manifold is symmetric with unstable manifold with
        # diagonal line y = x
        sManifold = 

        
        # get the spline interpolation of stable manifold. Note: unstable manifold are
        # double-valued, so we only interploate stable manifold, and this is
        # enough since unstable manifold and stable manifold is symmetric with line y = x.
        tck = splrep(sManifold[:,0], sManifold[:,1], s=0)

        # use scipy.optimize.fsolve() to obtain intersection points B, C, D
        # hint: use scipy.interpolate.splev() and the fact that stable and unstable
        # are symmetric with y = x
        C = 
        D = 
        B = 
        
        # save the variables needed for case3
        # if you are using ipython enviroment, you could just keep the varibles in the 
        # current session.
        np.savez('case2', B=B, C=C, D=D, eq0=eq0, eq1=eq1, 
                 sManifold=sManifold, uManifold=uManifold, tck=tck)
        
        # plot the unstable, stable manifold, points B, C, D, equilibria '0' and '1'.
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r-', lw=2, label=r'$W_u$')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c-', lw=2, label=r'$W_s$')
        ax.scatter(eq0[0],eq0[1])
        ax.scatter(eq1[0],eq1[1])
        ax.text(C[0], C[1], 'C')
        ax.text(D[0], D[1], 'D')
        ax.text(B[0], B[1], 'B')
        ax.text(eq0[0], eq0[1], '0')
        ax.text(eq1[0], eq1[1], '1')
        plt.legend()
        plt.show()

        
        
    if case == 3:  
        """
        Try to establish the first level partition of the non-wandering set
        in Henon map. You are going to iterate region 0BCD forward and backward
        for one step.
        """
        henon = Henon() # use the default parameters: a=6, b=-1
        # load needed variables from case2. If you have kept the variables
        # in the working space, then just comment out the following few lines.
        case2 = np.load('case2.npz')
        B = case2['B']; C = case2['C']; D = case2['D']; eq0 = case2['eq0']; eq1 = case2['eq1'];
        tck = case2['tck']; uManifold = case2['uManifold']; sManifold = case2['sManifold']; 
        
        # We first make a sampling of region 0BCD.
        # It works like this:
        # we are sure that box [-0.8, 0.8] x [-0.8, 0.8]
        # can cover region OBCD, but not every point inside this box is in 0BCD, 
        # so for point (x, y), how to determine whether it is in 0BCD ?
        # The criteria is
        #            y < f(x)  and  x < f(y)
        # Here f() is the interpolation function of stable manifold.
        # It is easy to see that 'y < f(x)' enforces point (x, y) below the
        # stable manifold, but the fact that 'x < f(y)' enforces point (x, y) to be
        # at the left side of the unstable manifold is a little tricky. The answer
        # is that stable manifold and unstable manifold are symmetric with y = x
        # Anyway, this part is implemented for you. 
        M = np.array([]).reshape(0,2) # region 0BCD
        x = np.arange(-0.8, 0.8, 0.01)
        y = splev(x, tck)
        for i in range(np.size(x)):
            for j in range(np.size(x)):
                if x[i] < y[j] and x[j] < y[i]:
                    state = np.array([x[i], x[j]])
                    M = np.vstack( (M, state) )

        # please plot out region M to convince yourself that you get region 0BCD
        # Now iterate forward and backward the points in region 0BCD for one step 
        Mf1 = # forward iteration points
        Mb1 = # backward iteration points
        
        # plot out Mf1 and Mb1
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(Mb1[:,0], Mb1[:,1], 'g.')
        ax.plot(Mf1[:,0], Mf1[:,1], 'm.')
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c')
        plt.show()

        # In order to see the pre-images of the boarders of Mf1 and Mb1, please
        # try to plot the images and per-images of 4 edges of region 0BCD.
        # hint: use the interpolation function of stable manifold
        
        
    if case == 4:
        """
        We go further into the partition of state space in this case.
        In case3 you have figure out what the pre-images of the border of 
        first forward and backward iteration, so we do not need to 
        sample the region again, iteration of the boarder is enough.
        In this case we iterate forward and backward for two steps
        """
        henon = Henon() # use the default parameters: a=6, b=-1
        # load needed variables from case2
        case2 = np.load('case2.npz')
        B = case2['B']; C = case2['C']; D = case2['D']; eq0 = case2['eq0']; eq1 = case2['eq1'];
        tck = case2['tck']; uManifold = case2['uManifold']; sManifold = case2['sManifold']; 
        
        # initialize the first/second forward/backward iteration of the boarder
        Mf1 = np.array([]).reshape(0,2) # the first forward iteration of the boarder you got in case 3
        Mf2 = np.array([]).reshape(0,2) # ... second forward ....
        Mb1 = np.array([]).reshape(0,2) # ....first backward ....
        Mb2 = np.array([]).reshape(0,2) # ....second backward ....

        # implement your code here to get Mf1, Mf2, Mb1, Mb2
        # hint: use the interpolation function of stable manifold   

        
        # plot out your result.
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c')
        ax.plot(Mf1[:,0], Mf1[:,1], 'm')
        ax.plot(Mf2[:,0], Mf2[:,1], 'g')
        ax.plot(Mb1[:,0], Mb1[:,1], 'b')
        ax.plot(Mb2[:,0], Mb2[:,1], 'y')
        ax.set_title('(d)')
        plt.show()
        
        # find a point in the top left region (the region which is closest to point D)
        # as the initial condition to find a periodic period with period 4
        # hint: use fsolve()
        guess = np.array([-0.4, 0.5])
        x = # the initial condition you get from this guess
        print henon.multiIter(x, 4) # check whether it is periodic
        # if you like, you can figure out the symbolic representation
        # of this periodic orbit.
        
