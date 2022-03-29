from matplotlib.pyplot import show
from numpy             import arange, array
from scipy.integrate   import odeint
from twoModes          import MultiPlotter, velocity


def get_orbit(x0,period,dtau = 0.005):
   return odeint(velocity, x0, arange(0, period, dtau))

if __name__ == '__main__':
   orbit1 = get_orbit(array([0.4525719, 0.0, 0.0509257, 0.0335428]), 3.6415120)
   orbit2 = get_orbit(array([0.4517771, 0.0, 0.0202026, 0.0405222]), 7.3459412)
   orbit3 = get_orbit(array([0.4514665, 0.0, 0.0108291, 0.0424373]), 14.6795175)
   orbit4 = get_orbit(array([0.4503967, 0.0, -0.0170958, 0.0476009]), 18.3874094)
   with MultiPlotter() as plotter:
      plotter.plot(orbit1[:,0:3])
      plotter.plot(orbit2[:,0:3])
      plotter.plot(orbit3[:,0:3])
      plotter.plot(orbit4[:,0:3])
   show()
