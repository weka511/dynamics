#!/usr/bin/env python
'''
   Relative periodic orbits from Table 12.1
'''

from argparse import ArgumentParser
import numpy as np
from scipy.integrate import odeint
from twoModes import MultiPlotter, reduceSymmetry, velocity, velocity_reduced

def get_orbit(x0, period,
              dtau = 0.005,
              n = 1):
   '''
   Integrate ODE to determine orbit

   Parameters:
       x0      Initial value
       period  Integrate for n repeats of orbit: this parameter define basic interval
       dtau    Time step
       n       Number of repeats of orbit
   '''
   return odeint(velocity, x0, np.arange(0, n*period, dtau))

def get_orbit_reduced(x0, period,
                      dtau = 0.005,
                      n = 1):
   '''
   Reduce symmetry, then integrate ODE to determine orbit

   Parameters:
       x0      Initial value
       period  Integrate for n repeats of orbit: this parameter define basic interval
       dtau    Time step
       n       Number of repeats of orbit
   '''
   return odeint(velocity_reduced, reduceSymmetry(x0), np.arange(0, n*period, dtau))

def parse_args():
   parser = ArgumentParser(__doc__)
   parser.add_argument('--n', default = 1, type = int, help = 'Number of repeats of orbit')
   return parser.parse_args()

if __name__ == '__main__':
   args = parse_args()

   with MultiPlotter(name='periodic') as plt1,   \
        MultiPlotter(name='reduced')  as plt2,   \
        MultiPlotter(name='inslice')  as plt3:

      orbit1   = get_orbit(np.array([0.4525719, 0.0, 0.0509257, 0.0335428]),   3.6415120, n = args.n)
      orbit2   = get_orbit(np.array([0.4517771, 0.0, 0.0202026, 0.0405222]),   7.3459412, n = args.n)
      orbit3   = get_orbit(np.array([0.4514665, 0.0, 0.0108291, 0.0424373]),  14.6795175, n = args.n)
      orbit4   = get_orbit(np.array([0.4503967, 0.0, -0.0170958, 0.0476009]), 18.3874094, n = args.n)

      inslice1 = get_orbit_reduced(np.array([0.4525719, 0.0, 0.0509257, 0.0335428]),   3.6415120, n = args.n)
      inslice2 = get_orbit_reduced(np.array([0.4517771, 0.0, 0.0202026, 0.0405222]),   7.3459412, n = args.n)
      inslice3 = get_orbit_reduced(np.array([0.4514665, 0.0, 0.0108291, 0.0424373]),  14.6795175, n = args.n)
      inslice4 = get_orbit_reduced(np.array([0.4503967, 0.0, -0.0170958, 0.0476009]), 18.3874094, n = args.n)

      plt1.plot(orbit1[:,0:3], title = '1')
      plt1.plot(orbit2[:,0:3], title = '01')
      plt1.plot(orbit3[:,0:3], title = '0111')
      plt1.plot(orbit4[:,0:3], title = '01101')

      plt2.plot(reduceSymmetry(orbit1)[:,0:3], title = '1')
      plt2.plot(reduceSymmetry(orbit2)[:,0:3], title = '01')
      plt2.plot(reduceSymmetry(orbit3)[:,0:3], title = '0111')
      plt2.plot(reduceSymmetry(orbit4)[:,0:3], title = '01101')

      plt3.plot(inslice1, title = '1')
      plt3.plot(inslice2, title = '01')
      plt3.plot(inslice3, title = '01111')
      plt3.plot(inslice4, title = '01101')
