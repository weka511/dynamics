##################################################
# This is the template file for you to investigate
# the dynamics in the anti-symmetric invariant
# subspace in Kuramoto-Sivashinsky system
##################################################
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.random import rand
from numpy.linalg import eig
from numpy.linalg import norm
from numpy.fft import fft, ifft
from scipy.integrate import  odeint
from scipy.optimize import fsolve

        
class KS:
    def __init__(self, d):
        """
        set the domain size L. we use d to denote it.
        """
        self.d = d
        
    def velocity(self, a, t):
        """
        input:
             a  state point [1xN] vector
             t  not used
        return velocity at state piont a
        Hint: np.convolve()
        """
        N = len(a)
        k = 2*np.pi/self.d * np.arange(1, N+1)
        L = k**2 -  k**4
        
        vel = 
        
        return vel

    def stabMat(self, a):
        """
        Stability matrix at point a
        
        """
        N = len(a)
        stab = 

        return stab
        

    def integrate(self, a0, h, nstp):
        """
        integrator is implemented for you.
        """
        aa = odeint(self.velocity, a0, np.arange(0, nstp*h, h))
        return aa

    
    def energy(self, aa):
        """
        return the energy at state points aa
        aa: states. Maybe a single state point or a sequence of points

        Note: Since we the system is anti-symmetric, we only care about
        the energy of half domain.
        Example: aa = np.narange(16)*0.1 => Energy is 6.2        
        """
        
        retun E

    def config(self, a):
        """
        This function is used to tranform the a state point in the
        Fourier space to the state in the configuration space.
        Already implemented for you.
        """
        N = a.shape[0]
        ap = np.zeros(2*N+1)
        ap[1:N+1] = a
        ap[N+1:] = -a[::-1]
        u = ifft(1j*ap).real
        
        return u[1:N+1]
    
    def configM(self, aa):
        M, N = aa.shape
        uu = zeros((M, N))
        for i in range(M):
            uu[i,:] = self.config(aa[i,:])
    
        return uu

    def plotConfig(self, uu):
        """
        Plot the states in the configuration space.
        """
        fig = plt.figure(figsize=(2,10))
        ax = fig.add_subplot(111)
        ax.imshow(uu, aspect='auto')
        #ax.set_xlabel('L')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        plt.tight_layout(pad=0)
        plt.show()
        




if __name__ == "__main__" :

    case = 1

    if case == 1:
        """
        validate implementation
        """
        # for domain size 36.33
        ks = KS(36.33)
        a0 = rand(16)*0.1
        h = 0.04
        aa = ks.integrate(a0, h, 5000); a0 = aa[-1, :]
        aa = ks.integrate(a0, h, 10000);
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(aa[:, 0], aa[:, 1], aa[:, 2])
        ax.set_xlabel(r'$a_1$')
        ax.set_ylabel(r'$a_2$')
        ax.set_zlabel(r'$a_3$')
        plt.tight_layout(pad=0)
        plt.show()

        # for domain size 38.5
        ks = KS(38.5)
        aa = ks.integrate(a0, h, 10000);
        uu= ks.configM(aa)
        ks.plotConfig(uu)
          
        # print velocity
        # it should print out
        #[ -1.82783573  -3.25773772  -4.29160436  -4.94992058  -5.28027032
        #  -5.36584904  -5.33397631  -5.36460822  -5.69884996  -6.64746834
        #  -8.59940423 -12.03028516 -17.51093779 -25.71590041 -37.4319355
        # -53.56654219]

        ks = KS(38.5)
        a0 = np.arange(16)*0.1
        print ks.velocity(a0, None)

        # calculate the average energe for d = 36.23
        # please finish this experiment
        ks = KS(36.23)
                
    
    if case == 2:
        """
        Try to obtain bifurcation tree for L in range [36.20, 36.40]
        Record the energe on Poincare section a1 = 0
        """
        
    
    if case == 3:
        """
        Try to have a look at the stable/unstable manifold of 
        one stable equilibrium and one unstable equilibrium.
        Also calculate the stability exponents.
        The initial conditions for these 2 equilbria is contained
        in file "eqL3633.npz". Also it contains a 16x16 matrix A0, 
        which is the stability matrix at point a0 = np.arange(16)*0.1
        to help you validate your implementation.
        """
        d = 36.33
        ks = KS(d)
        
        data = np.load('eqL3633.npz')
        eq1 = data['eq1']
        eq2 = data['eq2']
        A0 = data['A0']
        
