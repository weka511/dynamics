# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

import mol_dynamics as md,matplotlib.pyplot as plt,math

def get_count(N,R):
    try:
        c,_ = md.create_configuration(N=N,R=R,NT=1000,E=1,L=[1,1,1],D=3)
        return c
    except md.MolecularDynamicsError as e:
        return -1

params = sorted([(N,R,md.get_rho(N,R,[1,1,1])) for N in [25,50,100,130] for R in [0.015625, 0.03125, 0.0625]],
                key = lambda x:x[2])

colours = ['r','g','b']

for colour in colours:
    counts = [(rho,get_count(N,R)) for N,R,rho in params]
    plt.plot([rho for (rho,c) in counts if c>0],[math.log(c) for (_,c) in counts if c>0],c=colour)

plt.xlabel('rho')
plt.ylabel('log(number of attempts)')
plt.show()
