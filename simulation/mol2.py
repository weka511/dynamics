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

import random, math, numpy as np
from enum import Enum, unique

# boltzmann
#
# Bolzmann's distribution (to within a multiplicative constant, which will be lost when we scale)

def boltzmann(E,kT=1):
    return math.sqrt(E) * math.exp(-E/kT)
    
# MolecularDynamicsError
#
# An exception detected by the program itself

class MolecularDynamicsError(Exception):
    def __init__(self, message):
        self.message = message

# Particle
#
# One particle in the simulation

class Particle:
    def __init__(self,position=[0,0,0],velocity=[1,1,1],radius=1,m=1):
        self.position = [p for p in position]
        self.velocity = [v for v in velocity]
        self.radius   = radius
        self.events   = {}
        self.m        = m
        
    def __str__(self):
        return f'({self.position[0],self.position[1],self.position[2]}),({self.velocity[0]},{self.velocity[1]},{self.velocity[2]})'
    
    # get_distance2
    #
    # Squared distance between centres for some other Particle
    
    def get_distance2(self,other):
        return sum((self.position[i]-other.position[i])**2 for i in range(len(self.position)))
    
    # get_energy
    #
    # Determine kinetic energy of particle
    
    def get_energy(self):
        return 0.5*self.m*sum(v**2 for v in self.velocity)
    
    # scale_energy
    #
    # Used to adjust total energy by changing energy of each individual particle
    
    def scale_energy(self,energy_scale_factor):
        velocity_scale_factor = math.sqrt(energy_scale_factor)
        self.velocity         = [velocity_scale_factor*v for v in self.velocity]
    
    # reverse
    #
    # used when particle hits a wall to reverse perpendiclar component of velocity
    
    def reverse(self,index):
        self.velocity[index] = - self.velocity[index]
     
    # evolve
    # 
    # Change position bt applying veocity for specified time
    
    def evolve(self,dt):
        for i in range(len(self.position)):
            self.position[i] += (self.velocity[i]*dt)
        

# Wall
#
# This class represents walls of box
@unique
class Wall(Enum):
    NORTH  =  1
    EAST   =  2
    TOP    =  3
    SOUTH  = -NORTH
    WEST   = -EAST
    BOTTOM = -TOP
    
    def number(self):
        return abs(self._value_)-1
    
    @classmethod
    def get_wall_pair(self,index):
        if index==0:
            return(Wall.SOUTH,Wall.NORTH)
        elif index==1:
            return(Wall.WEST,Wall.EAST)
        elif index==2:
            return (Wall.BOTTOM,Wall.TOP)
 
# get_rho
#
# Density of particles
def get_rho(N,R,L,D=3):
    return (N*(4/D)*math.pi*R**D)/(L[0]*L[1]*L[2])
    
# create_configuration
#
# Build particles for box particles
# Make sure that spheres don't overlap 
def create_configuration(N=100,R=0.0625,NT=25,E=1,L=1,D=3):
    def get_position():
        return [random.uniform(R-L0,L0-R) for L0 in L]
    
    # get_velocity
    def get_velocity(): #Krauth Algorithm 1.21
        velocities = [random.gauss(0, 1) for _ in range(D)]
        sigma      = math.sqrt(sum([v**2 for v in velocities]))
        upsilon    = random.random()**(1/D)
        return [v*upsilon/sigma for v in velocities]
    
    # is_valid
    #
    # Make sure that spheres don't overlap 
    def is_valid(configuration):
        if len(configuration)==0: return False
        for i in range(N):
            for j in range(i+1,N):
                if configuration[i].get_distance2(configuration[j])<(2*R)**2:
                    return False
        return True
    
    product= []   # for create_configuration

    for i in range(NT):
        if is_valid(product):
            for particle in product:
                particle.velocity = get_velocity()
                
            actual_energy = sum([particle.get_energy() for particle in product])   
            for particle in product:
                particle.scale_energy(E/actual_energy)
            print (f'Radius = {R}, Density = {get_rho(N,R,L)}, {i+1} attempts')
            return i,product
        
        product= [Particle(position=get_position(),radius=R) for _ in range(N)]
        
    raise MolecularDynamicsError(f'Failed to create valid configuration for R={R}, density={get_rho(N,R,L)}, within {NT} attempts')

def get_next_pair_collision(configuration,t=0,R=0.0625,L=[1,1,1]):
    return math.inf,-1,-1

def get_next_wall_collision(configuration,t=0,R=0.0625,L=[1,1,1]):
    # get_collision
    #
    # Get collisions between particle and specified pair of opposite walls
    def get_collision_with_2walls(particle,index):
        wall_positive,wall_negative = Wall.get_wall_pair(index)
        coordinate                  = particle.position[index]
        assert abs(coordinate)<=L[index]
        velocity                    = particle.velocity[index]
    
        if velocity >0:        
            return t + (L[index]-R-coordinate)/velocity,wall_positive
        elif velocity <0:
            return t + (-L[index]+R+coordinate)/velocity,wall_negative
        else:                     # Collision will never happen
            return math.inf,None  
          
    t_best,j_best,wall_best = math.inf,None,None
    for j in range(len(configuration)):
        for index in range(3):                                        #FIXME
            t0,wall = get_collision_with_2walls(configuration[j],index)
            if t0<t_best:
                t_best,j_best,wall_best  = t0,j,wall
    return t_best,j_best,wall_best 

def wall_collision(j,wall):  
    particle = configuration[j]
    particle.reverse(wall.number())    

def pair_collision(k,l):
    pass

if __name__ == '__main__':
    import argparse,sys,matplotlib.pyplot as plt,time

    kT = 1 # Temperature in "Boltzmann units"
    
    def get_L(args_L):
        if type(args_L)==float:
            return [args_L,args_L,args_L]
        elif len(args_L)==1:
            return [args_L[0],args_L[0],args_L[0]]
        elif len(args_L)==3:
            return args_L
        else:
            print ('--L should have length 1 or 3')
            sys.exit(1)
         
        if len([l for l in L if l<=0]):
            print ('--L should be strictly positive')
            sys.exit(1)
            
    parser = argparse.ArgumentParser('Molecular Dynamice simulation')
    parser.add_argument('--N','-N', type=int,   default=25,      help='Number of particles')
    parser.add_argument('--T','-T', type=float, default=100,     help='Maximum Time')
    parser.add_argument('--R','-R', type=float, default=0.0625,  help='Radius of spheres')
    parser.add_argument('--NT',     type=int,   default=100,     help='Number of attempts to choose initial configuration')
    parser.add_argument('--NC',     type=int,   default=0,       help='Minimum number of collisions')
    parser.add_argument('--E',      type=float, default=1,       help='Total energy')
    parser.add_argument('--L',      type=float, default=1.0,     help='Half widths of box: one value or three.', nargs='+',)
    parser.add_argument('--seed',   type=int,   default=None,    help='Seed for random number generator')
    parser.add_argument('--freq',   type=int,   default=100,     help='Frequency: number of steps between progress reports')
    parser.add_argument('--show',               default=False,   help='Show plots at end of run',  action='store_true')
    parser.add_argument('--plots',              default='plots', help='Name of file to store plots')
    args = parser.parse_args()
    
    L    = get_L(args.L)
    
    if args.seed!=None:
        random.seed(args.seed)
        
    start_time = time.time()
    _,configuration = create_configuration(N=args.N, R=args.R, NT=args.NT, E=args.E, L=L )

    
    init_time = time.time()
    print (f'Time to initialize: {(init_time-start_time):.1f} seconds')
    t            = 0
    step_counter = 0
    collisions   = 0   # Number of particle-particle collisions - walls not counted   
    
    while t < args.T or collisions < args.NC:
        t_pair,k,l    = get_next_pair_collision(configuration,t=t,R=args.R,L=L)
        t_wall,j,wall = get_next_wall_collision(configuration,t=t,R=args.R,L=L)
        t_next        = min(t_pair,t_wall)
        if t_next == math.inf: break
        
        for particle in configuration:
            particle.evolve(t_next-t)
            
        if t_wall<t_pair:
            wall_collision(j,wall)
            print (t,t_next,j,wall)
        else:
            pair_collision(k,l)
            collision+=1
        t = t_next
            
    end_time = time.time()
    print (f'Time to simulate {collisions} collisions between {args.N} particles: {(end_time-init_time):.1f} seconds') 
    print (f'Total Time: {(end_time-start_time):.1f} seconds')    