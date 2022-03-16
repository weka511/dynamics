############################################################
# This file contains the related function to investigate
# the natural measure in Henon map.
#
# First, set case = 1 to plot the measure, you just need
# to fill out one line.
# Then, set case = 2 to locate the limit cycle
############################################################
from argparse             import ArgumentParser
from numpy                import array, size, zeros, histogram2d
from matplotlib.pyplot    import figure, show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors    import LogNorm
from numpy.random         import rand


class Henon:
    def __init__(self, a=1.4, b=0.3):
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
        return array([1 - self.a * x**2 + self.b * y, x])

    def multiIter(self, stateVec, NumOfIter):
        """
        forward iterate for multiple steps.
        stateVec: the current state. dimension : [1 x 2] numpy.array
        NumOfIter: number of iterations

        return: the current state and the furture 'NumOfIter' states.
                dimension [NumOfIter+1 x 2]
        """
        states = zeros((NumOfIter+1, 2))
        states[0,:] = stateVec
        tmpState = stateVec
        for i in range(NumOfIter):
            tmpState = self.oneIter(tmpState)
            states[i+1] = tmpState

        return states

    def naturalMeasure(self, initState, NumOfIter, nbins):
        """
        try to obtain the coarse-grained measure
        """
        states = self.multiIter(initState, NumOfIter)
        counts, xedges, yedges = \
            histogram2d(states[:,0], states[:,1], bins = nbins)
        return counts, xedges, yedges


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('case',
                    type    = int,
                    choices = [1,2])
    args = parser.parse_args()
    TBD = 0
    if args.case == 1:
        """
        plot the natural measure for a = 1.4, b = 0.3
        by long-time simulation.
        """
        henon = Henon(1.4, 0.3)
        counts, xedges, yedges = henon.naturalMeasure(rand(2), 1000000, 200)
        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]

        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x', size='x-large')
        ax.set_ylabel('y', size='x-large')
        ax.set_zlabel('count', size='x-large')
        for i in range(size(xedges) - 1):
            for j in range(size(yedges) - 1):
                if counts[i,j] != 0 :
                    x = xedges[i] + dx/2
                    y = yedges[j] + dy/2
                    z = TBD
                    ax.plot([x,x], [y,y], [0,z], 'b' )
        ax.set_title('(c)')
        show()

    if args.case == 2:
        """
        find the limit cycle for parameter a = 1.39945219, b = 0.3
        """
        henon = Henon(1.39945219, 0.3)
        # start from a random point, then evolve the system for a trancient period
        # discard the transcient part, plot the sequence that follows.


