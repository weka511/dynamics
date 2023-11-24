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

'''Exercise 9.2 Trapped orbits. Construct figure 1.9 using Monte-Carlo simulation'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from numpy.random import default_rng
from matplotlib.pyplot import figure, show
from pinball import Create_Centres, generate, create_pt, p_to_velocity

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-R', '--R', type=float, default=6.0, help='Centre to Centre distance')
    parser.add_argument('-a', '--a', type=float, default=1.0, help='Radius of each Disk')
    parser.add_argument('--N', default=100000, type=int,help='Number of trajectories')
    parser.add_argument('--seed', default=42, type=int,help='Initialize random number generator')
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def create_starting_values(N,rng = default_rng(),a=1):
    Starts = rng.random((args.N,2))
    return Starts*np.array([2*np.pi*a,2]) - np.array([0,1])

def monte_carlo(N,rng = default_rng(),a=1,Centres=None):
    for i in range(N):
        s = 2 * np.pi * rng.random()
        p = 2*rng.random() - 1
        count = 0
        for _,_,_,_,_,_ in generate(create_pt(s,radius=a,Centre=Centres[0]),
                                    p_to_velocity(p), Centres, a = a):
            count += 1
        if count > 0:
            yield s,p,count

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)

    ss = []
    ps = []
    counts = []
    for s,p,count in monte_carlo(args.N,rng = default_rng(args.seed),a=args.a,Centres = Create_Centres(args.R)):
        ss.append(s)
        ps.append(p)
        counts.append(count)
    ax.scatter(ss,ps,s=1,c=counts)
    ax.set_xlim(0,2*np.pi*args.a)
    ax.set_ylim(-1,1)
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
