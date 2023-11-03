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

'''Exercise 4.3.6'''


from argparse import ArgumentParser
from os.path import  basename,splitext
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

def f(theta,mu):
    return mu + np.sin(theta) + np.cos(2*theta)

if __name__=='__main__':
    start = time()
    args = parse_args()

    thetas = np.arange(0,2*np.pi,0.01)
    fig = figure(figsize=(10,10))
    for i,mu in enumerate([-0.1, 0,1,2,3]):
        ax = fig.add_subplot(2,4,i+1)
        ax.plot(thetas,f(thetas,mu),c='xkcd:blue')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\mu +\sin {\theta} + \cos {2 \theta}$')
        ax.set_title('$\mu=$' f'{mu}')
        ax.axhline(0,c='xkcd:red',linestyle=':')

    fig.suptitle(__doc__)
    fig.tight_layout()
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
