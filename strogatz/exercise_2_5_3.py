#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Exercise 2.5.3'''


from argparse import ArgumentParser
from os.path import join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_T(X,r=1,x0=0.1):
    def get_T_helper(X):
        return (np.log(X) - 0.5 * np.log(r + X**2))/r
    return get_T_helper(X) - get_T_helper(x0)

if __name__=='__main__':
    start  = time()
    args = parse_args()

    fig = figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    X = np.arange(0.1, 25, 0.05)
    for r in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        vget_T = np.vectorize(lambda X: get_T(X,r=r))
        ax.plot(X,vget_T(X),label=f'r={r}')
        ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('T')
    show()
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
