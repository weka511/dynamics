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

import heapq, random, abc,math
from enum import Enum, unique

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
        return sum((self.position[i]-other.position[i])**2 for i in range(3))
    
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
    # used when particle hits a wall to reverse perpendiclar compoenet of energy
    
    def reverse(self,index):
        self.velocity[index] = - self.velocity[index]
     
    # evolve
    # 
    # Change position bt applying veocity for specified time
    
    def evolve(self,dt):
        for i in range(3):
            self.position[i] += (self.velocity[i]*dt)
        

# Wall
#
# This class represents walls of box
@unique
class Wall(Enum):
    NORTH  =  1
    EAST   =  2
    SOUTH  = -1
    WEST   = -2
    TOP    =  3
    BOTTOM = -3
    
    def number(self):
        return abs(self._value_)-1
    
    @classmethod
    def get_wall_pair(self,index):
        if index==0:
            return(Wall.NORTH,Wall.SOUTH)
        elif index==1:
            return(Wall.EAST,Wall.WEST)
        elif index==2:
            return (Wall.TOP,Wall.BOTTOM)

# Event
#
# This class represents a collission between a particles and either another particle or a wall

class Event(abc.ABC):
    
    def __init__(self,t=math.inf):
        self.t = t
     
    # __lt__
    #
    # Enables us to store events in a priority queue, ordered by time
    
    def __lt__(self,other):
            return self.t<other.t 
    # act
    #
    # This is what happens during a collission
    
    @abc.abstractmethod
    def act(self,configuration,L=[1,1,1],R=0.0625,dt=0):
        pass

# HitsWall
#
# This class represents the event of a particle hitting a wall

class HitsWall(Event):
    def __init__(self,particle_index,wall,t=math.inf):
        super().__init__(t)
        self.particle_index = particle_index
        self.wall           = wall
        
    def __str__(self):
        return f'{self.t:.2f}\t\t({self.particle_index},{self.wall})'
    
    # act
    #
    # Reverse motion of particles perpendicular to wall
    
    def act(self,configuration,L=[1,1,1],R=0.0625,dt=0):
        super().act(configuration,L,R) 
        particle = configuration[self.particle_index]
        particle.reverse(self.wall.number())


# Collision
#
# This class represents a collision between two particles
class Collision(Event):
    def __init__(self,i,j,t=math.inf):
        super().__init__(t)
        assert i<j,f'Particle indices not in correct order -- we expect {i} < {j}'
        self.i = i     # Primary particle
        self.j = j     # Other particle: j>i
        
    def __str__(self):
        return f'{self.t:.2f}\t\t({self.i},{self.j})'

    # act
    #
    # Reverse motion along normal at point of collision
    
    def act(self,configuration,L=[1,1,1],R=0.0625,dt=0): #Krauth Algorithm 2.3
        super().act(configuration)
        particle_i          = configuration[self.i]
        particle_j          = configuration[self.j]
        delta_x             = [particle_i.position[k]-particle_j.position[k] for k in range(3)]
        delta_x_norm        = math.sqrt(sum(delta_x[k]**2 for k in range(3)))
        e_perp              = [delta_x[k]/delta_x_norm for k in range(3)]
        delta_v             = [particle_i.velocity[k]-particle_j.velocity[k] for k in range(3)]
        delta_v_e_perp      = sum(delta_v[k]*e_perp[k] for k in range(3))
        particle_i.velocity = [particle_i.velocity[k] - e_perp[k]*delta_v_e_perp for k in range(3)]
        particle_j.velocity = [particle_j.velocity[k] + e_perp[k]*delta_v_e_perp for k in range(3)]

# get_rho
#
# Density of particles
def get_rho(N,R,L):
    return (N*(4/3)*math.pi*R**3)/(L[0]*L[1]*L[2])
    
# create_configuration
#
# Build particles for box particles
# Make sure that spheres don't overlap 
def create_configuration(N=100,R=0.0625,NT=25,E=1,L=1):
    def get_position():
        return [random.uniform(R-l,l-R) for l in L]
    
    # get_velocity
    def get_velocity(d=3): #Krauth Algorithm 1.21
        velocities = [random.gauss(0, 1) for _ in range(d)]
        sigma      = math.sqrt(sum([v**2 for v in velocities]))
        upsilon    = random.random()**(1/d)
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
            return product
        
        product= [Particle(position=get_position(),radius=R) for _ in range(N)]
        
    raise MolecularDynamicsError(f'Failed to create valid configuration for R={R}, density={get_rho(N,R,L)}, within {NT} attempts')

# Build dictionaries of all possible events. We will extract active events as we compute collisions
def link_events(configuration):
    
    for i in range(len(configuration)):
        for wall in Wall:
            configuration[i].events[wall]=HitsWall(i,wall)
        for j in range(i+1,len(configuration)):
            configuration[i].events[j]= Collision(i,j)
 
# get_collisions_sphere_wall
#
# Get collisions between specified particle and any wall

def get_collisions_sphere_wall(particle,t=0,L=1,R=0.0625):
    # get_collision
    #
    # Get collisions between particle and specified pair of opposite walls
    def get_collision(index):
        direction_positive,direction_negative = Wall.get_wall_pair(index)
        distance                              = particle.position[index]
        velocity                              = particle.velocity[index]
        event_plus                            = particle.events[direction_positive]

        if velocity ==0:
            event_plus.t = float.inf
            return event_plus
        if velocity >0:        
            event_plus.t = t + (L[index]-R-distance)/velocity
            return event_plus
        if velocity <0:
            event_minus = particle.events[direction_negative]
            event_minus.t = t + (L[index]-R+distance)/abs(velocity) 
            return event_minus
        
    return [get_collision(index) for index in range(3)]

# get_collisions_sphere_sphere
#
# get all collisions between specifed particle and all others having index greater that specified particle

def get_collisions_sphere_sphere(i,configuration,t=0,R=0.0625):
    # get_next_collision
    # Determine whether two particles are on a path to collide
    # Krauth Algorithm 2.2
    def get_next_collision(particle1,particle2):  
        dx    = [particle1.position[k] - particle2.position[k] for k in range(3)]
        dv    = [particle1.velocity[k] - particle2.velocity[k] for k in range(3)]
        dx_dv = sum(dx[k]*dv[k] for k in range(3))
        dx_2  = sum(dx[k]*dx[k] for k in range(3))
        dv_2  = sum(dv[k]*dv[k] for k in range(3))
        disc  = dx_dv**2 - dv_2 * (dx_2 -R**2)
        if disc>=0 and dx_dv<0:
            return (-dx_dv + math.sqrt(disc))/dv_2

    # get_collision_with
    #
    # Get possible collision between particl and specified other particle
    def get_collision_with(j):
        dt = get_next_collision(configuration[i],configuration[j])
        if dt !=None:
            collision = configuration[i].events[j]
            collision.t = t + dt
            return collision
    
    #  non_trivial
    #
    # Purge Nones from list
    def non_trivial(list):
        return [l for l in list if l!=None]
    
    return non_trivial([get_collision_with(j) for j in range(i+1,len(configuration))])

# flatten
#
# Flatten a list of lists into a simple list

def flatten(lists):
    return [item for sublist in lists for item in sublist]

if __name__ == '__main__':
    import argparse,sys,matplotlib.pyplot as plt
    
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
    parser.add_argument('--N','-N', type=int,   default=25,             help='Number of particles')
    parser.add_argument('--T','-T', type=float, default=100,            help='Maximum Time')
    parser.add_argument('--R','-R', type=float, default=0.0625,         help='Radius of spheres')
    parser.add_argument('--NT',     type=int,   default=100,            help='Number of attempts to choose initial configuration')
    parser.add_argument('--E',      type=float, default=1,              help='Total energy')
    parser.add_argument('--L',      type=float, default=1.0, nargs='+', help='Half widths of box: one value or three.')
    parser.add_argument('--seed',   type=int,   default=None,           help='Seed for random number generator')
    args = parser.parse_args()
    
    L    = get_L(args.L)
    
    if args.seed!=None:
        random.seed(args.seed)
        
    try:
        configuration = create_configuration(N=args.N, R=args.R, NT=args.NT, E=args.E, L=L )
        link_events(configuration)
        t  = 0
         
        while t < args.T:
            events = flatten([get_collisions_sphere_wall(configuration[i],t=t,L=L,R=args.R) for i in range(args.N)] + \
                             [get_collisions_sphere_sphere(i,configuration,t=t,R=args.R) for i in range(args.N)])
            
            heapq.heapify(events)
            next_event = events[0]
            dt         = next_event.t-t
            print (f'{next_event}')
            t          = next_event.t
            for particle in configuration:
                particle.evolve(dt)
            next_event.act(configuration,L=L,R=args.R,dt=dt)
        plt.figure(figsize=(20,10))
        plt.hist([particle.get_energy() for particle in configuration])
        plt.title(f'N={args.N}, T={args.T}, rho={get_rho(args.N,args.R,L):.2f}')
        plt.xlabel('Energy')
        plt.savefig('energies.png')
        plt.show()
    except MolecularDynamicsError as e:
        print (e)
        sys.exit(1)
    except:
        print(f'Unexpected error: {sys.exc_info()}')
        sys.exit(1)
 