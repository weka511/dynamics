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
    parser.add_argument('--seed', default=None, type=int,help='Initialize random number generator')
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-',figs='./figs'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def monte_carlo_generator(N,
                          rng = default_rng(),
                          a = 1,
                          Centres = None,
                          threshold = 1):
    '''
    Generate several starting points and explore trajectory. If more than specified number
    of bounces, yield value
    '''
    for i in range(N):
        s = 2 * np.pi * rng.random()
        p = 2*rng.random() - 1
        count = sum([1 for _ in generate(create_pt(s,radius=a,Centre=Centres[0]), p_to_velocity(p), Centres, a = a)])
        if count > threshold:
            yield s,p,count

def monte_carlo(N,a=1,R=6,rng = default_rng()):
    ss = []
    ps = []
    counts = []
    for s,p,count in monte_carlo_generator(N,rng = rng,a=a,Centres = Create_Centres(R)):
        ss.append(s)
        ps.append(p)
        counts.append(count)
    return ss,ps,counts

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)

    ss,ps,counts = monte_carlo(args.N,rng = default_rng(args.seed),a=args.a,R=args.R)
    sss = []
    pss = []
    for i in range(len(ss)):
        if counts[i] > 2:
            sss.append(ss[i])
            pss.append(ps[i])
    ax1.scatter(ss,ps,s=1,c=counts)

    ax1.set_xlim(0,2*np.pi*args.a)
    ax1.set_ylim(-1,1)
    ax1.set_title('At least one bounce')
    ax1.set_xlabel('s')
    ax1.set_ylabel('p')

    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(sss,pss,s=1)
    ax2.set_xlim(0,2*np.pi*args.a)
    ax2.set_ylim(-1,1)
    ax2.set_title('At least two bounces')
    ax2.set_xlabel('s')
    ax2.set_ylabel('p')

    fig.suptitle(f'{args.N:,} Iterations')
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
