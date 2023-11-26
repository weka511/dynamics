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
from pinball import Create_Centres, generate, create_pt, get_velocity
from xkcd import generate_colours
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-R', '--R', type=float, default=2.5, help='Centre to Centre distance')
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

    Parameters:
        N           Number of trajectories
        rng
        a           Radius of each Disk
        Centres
        threshold
    '''
    for i in range(N):
        s = np.pi*(2*rng.random() - 1)
        p = 2*rng.random() - 1
        v = get_velocity(s,p)#p_to_velocity(p)
        count = sum([1 for _ in generate(create_pt(s,radius=a,Centre=Centres[0]), v, Centres, a = a, first_bounce=0)])
        if count > threshold:
            yield s,p,count


def monte_carlo(N,a=1,R=6,rng = default_rng()):
    '''
    Initialize disks and perform Monte Carlo simulation

    Parameters:
        N      Number of trajectories
        a      Radius of each Disk
        R      Centre to Centre distance
        rng    Random number generator

    Returns:
        List of values for s for starting value
        List of values for p for starting value
        List containing length of each orbit
    '''
    values= np.array([x for x in monte_carlo_generator(N,rng = rng,a = a,Centres = Create_Centres(R))])
    return values[values[:,2].argsort()]
    # ss = []
    # ps = []
    # orbit_lengths = []
    # for s,p,count in monte_carlo_generator(N,rng = rng,a = a,Centres = Create_Centres(R)):
        # ss.append(s*R/(2*np.pi))
        # ps.append(p)
        # orbit_lengths.append(count)
    # return ss,ps,orbit_lengths

# def prune(s,p,orbit_lengths,threshold=2):
    # '''
    # Remove starting values whose length of trajectory fails to exceed thrshold

    # Parameters:
        # s              List of values for s for starting value
        # p              List of values for s for starting value
        # orbit_lengths  List containing length of each orbit
        # threshold      We discard any orbits for which length doesn't execeed threshold

    # Returns:
        # List of values for s for starting value (pruned values only)
        # List of values for p for starting value (pruned values only)
        # List containing length of each orbit    (pruned values only)
    # '''
    # pruned = np.array([(s[i],p[i], orbit_lengths[i]) for i in range(len(s)) if orbit_lengths[i] > threshold])
    # return pruned[pruned[:,2].argsort()]

# def get_colours(ns):
    # colours = []
    # def get_colour(n):
        # if n==2:
            # return 'xkcd:green'
        # if n==3:
            # return 'xkcd:blue'
        # return 'xkcd:red'
    # return [get_colour(n) for n in ns]

if __name__=='__main__':
    start  = time()
    args = parse_args()

    starts_counts = monte_carlo(args.N,rng = default_rng(args.seed),a=args.a,R=args.R)
    # pruned = prune(s,p,orbit_lengths)
    n_colours = int(starts_counts[:,2].max())
    n,bins = np.histogram(starts_counts[:,2],bins = n_colours)
    cumulative_counts = np.cumsum(n[::-1])[::-1]
    survival_ratio = cumulative_counts[1:-1]/cumulative_counts[0:-2]
    gamma = - np.log(survival_ratio)

    fig = figure(figsize=(16,8))
    ax1 = fig.add_subplot(2,2,1)
    xkcd = []
    for i,c in enumerate(generate_colours()):
        xkcd.append(c)
        if i>n_colours: break
    colours = [xkcd[int(x)] for x in starts_counts[:,2]]
    ax1.scatter(starts_counts[:,0],starts_counts[:,1],s=1,c=colours)
    ax1.set_xlim(-args.R,args.R)
    ax1.set_ylim(-1,1)
    ax1.set_title(f'At least one bounce: n={starts_counts.shape[0]:,}')
    ax1.set_xlabel('s')
    ax1.set_ylabel('p')

    # ax2 = fig.add_subplot(2,2,2)
    # ax2.scatter(pruned[:,0],pruned[:,1],s=1,c=pruned[:,2])
    # ax2.set_xlim(-args.R,args.R)
    # ax2.set_ylim(-1,1)
    # ax2.set_title(f'At least two bounces: n={pruned.shape[0]:,}')
    # ax2.set_xlabel('s')
    # ax2.set_ylabel('p')

    ax3 = fig.add_subplot(2,2,3)
    ax3.bar(bins[:-1],n,width=1)
    ax3.set_title('Number of bounces')
    ax3.set_xlabel('n')
    ax3.set_ylabel('Frequency')

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(gamma)
    ax4.set_xlabel('n')
    ax4.set_ylabel(r'$\gamma_{n}')
    ax4.set_title(f'Escape Rate {np.mean(gamma[2:]):.4f}')
    fig.suptitle(f'{args.N:,} Iterations. R={args.R}, a={args.a}, seed={args.seed}')
    fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
