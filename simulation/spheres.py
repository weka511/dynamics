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

N = 25
Lx = 1.0
Ly = 1.0
Lz = 1.0
R  = 0.0625
M  = 25
E  = 100

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
    NORTH = 0
    EAST  = 1
    SOUTH = 2
    WEST  = 3
    FRONT = 4
    BACK  = 5
    
class Event(abc.ABC):
    def __init__(self,t=math.inf):
        self.t = t
    def __lt__(self,other):
            return self.t<other.t 
        
    @abc.abstractmethod
    def act(self):
        pass
    
class HitsWall(Event):
    def __init__(self,i,wall,t=math.inf):
        super().__init__(t)
        self.i    = i
        self.wall = wall
    def __str__(self):
        return f'({self.i},{self.wall})\t\t{self.t}'
    def act(self):
        super().act()    
    
class Collision(Event):
    def __init__(self,i,j,t=math.inf):
        super().__init__(t)
        self.i = i
        self.j = j
        
    def __str__(self):
        return f'({self.i},{self.j})\t\t\t{self.t}'
    
    def act(self):
        super().act()       
      
# Build particles for box particles
def create_configuration():
    def get_position():
        return [random.uniform(-Lx,Lx),random.uniform(-Ly,Ly),random.uniform(-Lz,Lz)]
    def get_velocity(d=3): #Krauth Algorithm 1.21
        velocities = [random.gauss(0, 1) for _ in range(d)]
        sigma = math.sqrt(sum([v**2 for v in velocities]))
        upsilon = random.random()**(1/d)
        return [v*upsilon/sigma for v in velocities]
    def valid(configuration):
        if len(configuration)==0: return False
        for i in range(N):
            for j in range(i+1,N):
                if configuration[i].get_distance2(configuration[j])<R**2:
                    return False
        return True
    
    configuration= []

    for i in range(M):
        if valid(configuration):
            for particle in configuration:
                particle.velocity = get_velocity()
            E0 = sum([particle.get_energy() for particle in configuration])   
            for particle in configuration:
                particle.scale_energy(E/E0)
                #print (particle)
            return configuration
        configuration= [Particle(position=get_position(),radius=R) for _ in range(N)]
    raise Exception(f'Failed to create valid configuration for R={R} within {M} attempts')

# Build dictionaries of all possible events. We will extract active events as we compute collisions
def link_events():
    
    for i in range(N):
        for wall in Wall:
            particles[i].events[wall]=HitsWall(i,wall)
        for j in range(i+1,N):
            particles[i].events[j]= Collision(i,j)
    

if __name__ == '__main__':                
    particles = create_configuration()
    
    link_events()
    
    # calculate all possible collisions - make them into active events
    
    # find next collision
    
    # update all events associated with colliding spheres


        
#heapq.heapify(active_events)

#while (len(active_events)>0):
    #print(heapq.heappop(active_events))

#t = 0    
#while True:
    #next_events = heapq.heappop(events)
    #t = next_events.t
    #next_events.action()
    