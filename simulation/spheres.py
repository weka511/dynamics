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

L           = [1.0,1.0,1.0]   # dimensions of box

class Particle:
    def __init__(self,position=[0,0,0],velocity=[1,1,1],radius=1):
        self.position = [p for p in position]
        self.velocity = [v for v in velocity]
        self.radius   = radius
        self.events   = {}
    def __str__(self):
        return f'({self.position[0],self.position[1],self.position[2]}),({self.velocity[0]},{self.velocity[1]},{self.velocity[2]})'
    def get_distance2(self,other):
        return sum([(self.position[i]-other.position[i])**2 for i in range(3)])
    def get_energy(self):
        return sum([v**2 for v in self.velocity])
    def scale_energy(self,energy_scale_factor):
        velocity_scale_factor = math.sqrt(energy_scale_factor)
        self.velocity = [velocity_scale_factor*v for v in self.velocity]
        
@unique
class Wall(Enum):
    NORTH  = 0
    EAST   = 1
    SOUTH  = 2
    WEST   = 3
    TOP    = 4
    BOTTOM = 5
    
class Event(abc.ABC):
    def __init__(self,t=math.inf):
        self.t = t
        
    def __lt__(self,other):
            return self.t<other.t 
        
    @abc.abstractmethod
    def act(self,configuration):
        pass
    
class HitsWall(Event):
    def __init__(self,particle_index,wall,t=math.inf):
        super().__init__(t)
        self.particle_index = particle_index
        self.wall           = wall
    def __str__(self):
        return f'({self.particle_index},{self.wall})\t\t{self.t}'
    def act(self,configuration):
        super().act(configuration) 
        particle = configuration[self.particle_index]
        print (self)
        if self.wall==Wall.EAST or self.wall==Wall.WEST:
            particle.velocity[0] = -particle.velocity[0]
        if self.wall==Wall.NORTH or self.wall==Wall.SOUTH:
            particle.velocity[1] = -particle.velocity[1]
        if self.wall==Wall.TOP or self.wall==Wall.BOTTOM:
            particle.velocity[2] = -particle.velocity[2]                
    
class Collision(Event):
    def __init__(self,i,j,t=math.inf):
        super().__init__(t)
        self.i = i
        self.j = j
        
    def __str__(self):
        return f'({self.i},{self.j})\t\t\t{self.t}'
    
    def act(self,configuration):
        super().act(configuration)       
      
# Build particles for box particles
def create_configuration(N=100,R=0.0625,NT=25,E=1):
    def get_position():
        return [random.uniform(-l,l) for l in L]
    
    def get_velocity(d=3): #Krauth Algorithm 1.21
        velocities = [random.gauss(0, 1) for _ in range(d)]
        sigma      = math.sqrt(sum([v**2 for v in velocities]))
        upsilon    = random.random()**(1/d)
        return [v*upsilon/sigma for v in velocities]
    
    def is_valid(configuration):
        if len(configuration)==0: return False
        for i in range(N):
            for j in range(i+1,N):
                if configuration[i].get_distance2(configuration[j])<R**2:
                    return False
        return True
    
    product= []

    for i in range(NT):
        if is_valid(product):
            for particle in product:
                particle.velocity = get_velocity()
                
            actual_energy = sum([particle.get_energy() for particle in product])   
            for particle in product:
                particle.scale_energy(E/actual_energy)        
            return product
        
        product= [Particle(position=get_position(),radius=R) for _ in range(N)]
        
    raise Exception(f'Failed to create valid configuration for R={R} within {N_attempts} attempts')

# Build dictionaries of all possible events. We will extract active events as we compute collisions
def link_events(configuration):
    
    for i in range(len(configuration)):
        for wall in Wall:
            configuration[i].events[wall]=HitsWall(i,wall)
        for j in range(i+1,len(configuration)):
            configuration[i].events[j]= Collision(i,j)
    
def get_collisions_sphere_wall(i,configuration,t):
    def get_collision(direction_positive,direction_negative,index):
        particle    = configuration[i]
        distance    = particle.position[index]
        velocity    = particle.velocity[index]
        event_plus  = particle.events[direction_positive]

        if velocity ==0:
            event_plus.t = float.inf
            return event_plus
        if velocity >0:        
            event_plus.t = t + (L[index]-distance)/velocity
            return event_plus
        if velocity <0:
            event_minus = particle.events[direction_negative]
            event_minus.t = t + (L[index]+distance)/abs(velocity)      
            return event_minus
        
    return [get_collision(Wall.EAST,Wall.WEST,0),
            get_collision(Wall.NORTH,Wall.SOUTH,1),
            get_collision(Wall.TOP,Wall.BOTTOM,2)]

def get_collisions_sphere_sphere(i):
    return []

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Molecular Dynamice simulation')
    parser.add_argument('--N','-N', type=int,   default=25,     help='Number of particles')
    parser.add_argument('--T','-T', type=float, default=100,    help='Maximum Time')
    parser.add_argument('--R','-R', type=float, default=0.0625, help='Radius of spheres')
    parser.add_argument('--NT',     type=int,   default=100,    help='Number of attempts to choose initial configuration')
    parser.add_argument('--E',      type=float, default=1,      help='Total energy')
   
    args = parser.parse_args()
    
    configuration = create_configuration(N=args.N, R=args.R, NT=args.NT, E=args.E)
    link_events(configuration)
    t      = 0
     
    while t < args.T:
        print (f't={t:.2f}')
        event_lists = \
            [get_collisions_sphere_wall(i,configuration,t) for i in range(args.N)] +\
            [get_collisions_sphere_sphere(i) for i in range(args.N)] 
        events      =  [item for sublist in event_lists for item in sublist]
        heapq.heapify(events)
        next_event = events[0]
        t          = next_event.t
        next_event.act(configuration)
 