#!/usr/bin/env python

'''Plot areas and volumes of hyperspheres'''

from matplotlib.pyplot import legend, plot, savefig, show, title
from math import pi
from scipy.special import gamma

n  = 100
Ds = [d for d in range(2,n+1)]

S = [(2 * pi**(d/2))/gamma(d/2) for d in range(2,n+1)]
V = [(pi**(d/2))/gamma(d/2+1)  for d in range(2,n+1)]

plot(Ds, S, label = f'Surface Area: max at d={Ds[S.index(max(S))]}')
plot(Ds, V, label = f'Volume: max at d={Ds[V.index(max(V))]}')
legend()
title ('Areas and volumes of hyperspheres')
savefig('hypershere')
show()
