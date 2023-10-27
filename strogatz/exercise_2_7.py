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

'''Exercises 2.7.3 and 2.7.4'''


from argparse import ArgumentParser
from os.path import  basename,splitext
from time import time
import numpy as np
from matplotlib.pyplot import figure, show, rcParams

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_name_for_save():
    '''Extract name for saving figure'''
    return splitext(basename(__file__))[0]

if __name__=='__main__':
    rcParams['text.usetex'] = True
    start  = time()
    args = parse_args()
    xs = np.arange(-10,10,0.1)
    fig = figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1)
    V1 = np.cos(xs)
    ax1.plot(xs,V1,color='xkcd:blue',label='Potential')
    ax1.axhline(V1.min(),color='xkcd:green',linestyle='dotted',label='Stable')
    ax1.axhline(V1.max(),color='xkcd:red',linestyle='dotted',label='Unstable')
    ax1.set_xlabel('x')
    ax1.set_ylabel('V(x)')
    ax1.set_title('Exercise 2.7.3: $\\dot{x}=\\sin{x}$')
    ax1.legend()
    ax2 = fig.add_subplot(2,1,2)
    V2 = -2*xs + np.cos(xs)
    ax2.plot(xs,V2,color='xkcd:blue')
    ax2.set_xlabel('x')
    ax2.set_ylabel('V(x)')
    ax2.set_title('Exercise 2.7.4: $\\dot{x}=2+\\sin{x}$')
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
