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

'''
    Exercise 6.1.8 from Strogatz: Van der Pol
'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from  phase import generate, plot_phase_portrait, adapt

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    X,Y,U,V,fixed = generate(f = lambda x,y:(y,-x+y*(1-x**2)))
    plot_phase_portrait(X,Y,U,V,fixed,title = r'$\dot{x} = y,\dot{y} = -x + y(1-x^2)$',ax=ax)
    fig.suptitle(__doc__)
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
