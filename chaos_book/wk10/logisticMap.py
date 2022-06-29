###################################################
# This file contains all the related functions to
# investigate the escape rate in Logistic map.
# please complete the experiment part
###################################################
from argparse          import ArgumentParser
from numpy             import ones, log, mean, std, zeros
from matplotlib.pyplot import figure, legend, plot, savefig, semilogy, show, title
from random            import random
from scipy.stats       import linregress

class Logistic:
    def __init__(self, A):
        self.A = A

    def oneIter(self, x):
        return self.A * (1.0 - x) * x

    def multiIters(self, x, n):
        y    = x
        tmpx = x
        for i in range(n):
            tmpx = self.oneIter(tmpx)
            y.append(tmpx)

        return y

    def doesEscape(self, x, n):
        '''
        determine whether the mapping sequence is escaping or not
        parameters:
              x  initial point
              n  number of iteration
        return :
             a vector indicating whether the corresponding iteration
             has escapted region [0, 1] or not. '1' indicates escapted.
        '''
        tmpx   = x
        escape = ones(n)
        for i in range(n):
            tmpx = self.oneIter(tmpx)
            if tmpx <= 1 and tmpx >=0:
                escape[i] = 0
            else:
                break

        return escape


if __name__ == '__main__':
    parser = ArgumentParser('Q10.4 Escape rate in Logistic map (Exercise 20.2)')
    parser.add_argument('--A',
                        type    = float,
                        default = 6)
    parser.add_argument('--M',
                        type    = int,
                        default = 100000,
                        help = 'start with a large number of initial conditions')
    parser.add_argument('--N',
                        type    = int,
                        default = 14,
                        help = 'iterate for a certain number of steps')
    parser.add_argument('--K',
                        type    = int,
                        default = 5,
                        help = 'number of samples')
    args       = parser.parse_args()
    fig        = figure(figsize=(12,12))
    mapping    = Logistic(args.A)
    EscapeRate = []
    for k in range(args.K):
        N_escapes = zeros(args.N)
        for i in range(args.M):
            N_escapes +=  mapping.doesEscape(random(),args.N)

        Gamma       = (args.M-N_escapes)/args.M
        result      = linregress(range(args.N), log(Gamma))
        EscapeRate.append(-result.slope)
        plot (Gamma,label=f'Escape rate={-result.slope:.6f}, stderr={result.stderr:.6f}')
        semilogy()

    title(f'A={args.A}, Escape Rate = {mean(EscapeRate):.6f}, std={std(EscapeRate):.6f})')
    legend()
    savefig('Logistic')
    show()
