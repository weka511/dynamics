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

# This program models molecular dynamica, after Alder & Wainwright

import random, abc, math, numpy as np
from enum import Enum, unique

# boltzmann
#
# Bolzmann's distribution (to within a multiplicative constant, which will be lost when we scale)
#
# See https://en.wikipedia.org/wiki/Boltzmann_distribution
def boltzmann(E,kT=1):
    return  math.exp(-E/kT)
    
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
    # used when particle hits a wall to reverse perpendicular component of velocity
    
    def reverse(self,index,L=1): # FIXME (L) 
        self.velocity[index] = - self.velocity[index]
        self.position[index] = -L if self.velocity[index]>0 else +L   
     
    # evolve
    # 
    # Change position by applying velocity for specified time
    
    def evolve(self,dt):
        for i in range(len(self.position)):
            self.position[i] += (self.velocity[i]*dt)
        

# Wall
#
# This class represents boundary of space. Walls are organized into pairs, e.g. NORTH-SOUTH
@unique
class Wall(Enum):
    NORTH  =  1
    EAST   =  2
    TOP    =  3
    SOUTH  = -1
    WEST   = -2
    BOTTOM = -3
    
    # get_index
    #
    # Determine the index corresponding to a pair of walls
    def get_index(self):
        return abs(self._value_)-1

    # get_wall_pair
    #
    # Look up wall pair for specific index
    @classmethod
    def get_wall_pair(self,index):
        if index==0:
            return(Wall.NORTH,Wall.SOUTH)
        elif index==1:
            return(Wall.EAST,Wall.WEST)
        elif index==2:
            return (Wall.TOP,Wall.BOTTOM)

# This class models the topology; it can be either a simple box or a torus.
# This class decides how a particle behaves at the wall

class Topology(abc.ABC):
 
    def __init__(self,name,L,R):
        self.myName = name
        self.L      = L
        self.R      = R
    
    # hitsWall
    #
    # This class is called when a particle behaves at the wall
    @abc.abstractmethod
    def hitsWall(self,particle,index):
        pass
    
    # name
    #
    # This is the name as used to create the Topology
    def name(self):
        return self.myName
    
    # pretty
    #
    # Name for plot title
    
    def pretty(self):
        return self.myName[0].upper()+self.myName[1:].lower()

# Box
#
# This topology represent a closed box. Particles pass through walls.

class Box(Topology):
    
    def __init__(self,L,R):
        super().__init__("box",L,R)
    
    # hitsWall
    #
    # Reverse motion of particles perpendicular to wall
    
    def hitsWall(self,particle,index):
        particle.reverse(index)
        
# Torus
#
# In this toplogy, particle wraps around when it reaches boundary

class Torus(Topology):
    def __init__(self,L,R):
        super().__init__("torus",L,R)
        
    # wrap particle around by moving to other wall
    def hitsWall(self,particle,index):
        particle.position[index] = - particle.position[index]  
        
# Event
#
# This class represents a collision between a particles and either another particle or a wall
# It encapsulates time, which is zero when the simulation starts, and increases thereafter.

class Event(abc.ABC):
    
    def __init__(self,t_expected=math.inf):
        self.t_expected = t_expected   # Time when event is expected to occur
     
    # __lt__
    #
    # Enables us to store events in a priority queue, ordered by expected time
    
    def __lt__(self,other):
            return self.t_expected<other.t_expected 
    # act
    #
    # This is what happens during a collission
    
    @abc.abstractmethod
    def act(self,configuration,topology):
        pass
    
    # List of particles that are involved in the event.
    # Used to cull them so their next collision can be calculated.    
    @abc.abstractmethod
    def get_colliders(self):
        pass
    

# HitsWall
#
# This class represents the event of a particle hitting a wall

class HitsWall(Event):
    def __init__(self,particle_index,wall,t_expected=math.inf):
        super().__init__(t_expected)
        self.particle_index = particle_index
        self.wall           = wall
        
    def __str__(self):
        return f'{self.t_expected:.2f}\t\t({self.particle_index},{self.wall})'

        
    # get_collisions
    #
    # Get collisions between specified particle and any wall
    @classmethod
    def get_collisions(self,particle,t_simulated=0,L=1,R=0.0625):
        # get_collision
        #
        # Get collisions between particle and specified pair of opposite walls
        def get_collision(index):
            direction_positive,direction_negative = Wall.get_wall_pair(index)
            distance                              = particle.position[index]
            velocity                              = particle.velocity[index]
            event_plus                            = particle.events[direction_positive]
    
            if velocity ==0:                  # Collision will never happen
                event_plus.t_expected = float.inf      # We could return a None and tidy if up, 
                return event_plus             # but this is simpler
            if velocity >0:        
                event_plus.t_expected = t_simulated + (L[index]-R-distance)/velocity
                return event_plus
            if velocity <0:
                event_minus = particle.events[direction_negative]
                event_minus.t_expected = t_simulated + (L[index]-R+distance)/abs(velocity) 
                return event_minus
            
        return [get_collision(index) for index in range(len(particle.position))]  
    
    # act
    #
    # Reverse motion of particles perpendicular to wall
    
    def act(self,configuration,topology):
        super().act(configuration,topology) 
        particle = configuration[self.particle_index]
        index    = self.wall.get_index()
        # Collision should be at a distance of R from wall, butthis may be affected by roundoff
        # Fixup here
        if abs( particle.position[index])!=topology.L[index]-topology.R:
            particle.position[index] = math.copysign(topology.L[index]-topology.R, particle.position[index])
        topology.hitsWall(particle,index)
        return 0 # We don't want to count these collisions
    
    # List of particles that are involved in ths event.
    # Used to cull them so their next collision can be calculated.
    def get_colliders(self):
        return [self.particle_index]
    
# Collision
#
# This class represents a collision between two particles
class Collision(Event):
    def __init__(self,particle_index,other_particle,t=math.inf):
        super().__init__(t)
        assert particle_index<other_particle,\
               f'Particle indices not in correct order -- we expect {particle_index} < {other_particle}'
        self.particle_index = particle_index     # Primary particle
        self.other_particle = other_particle     # Other particle: j>i
        
    def __str__(self):
        return f'{self.t_expected:.2f}\t\t({self.particle_index},{self.other_particle})'
        
    # get_collisions
    #
    # get all collisions between specifed particle and all others having index greater that specified particle
    
    @classmethod
    def get_collisions(self,particle_index,configuration,t_simulated=0,R=0.0625):
  
        # get_collision_with
        #
        # Get possible collision between particle and specified other particle
        def get_collision_with(other):
            # get_next_collision
            # Determine whether two particles are on a path to collide.
            # If so, return expected time to next collision.
            # Krauth Algorithm 2.2
            def get_next_collision(particle1,particle2):
                dx    = [pos1 - pos2  for (pos1,pos2) in zip(particle1.position, particle2.position)]
                dv    = [vel1 - vel2  for (vel1,vel2) in zip(particle1.velocity,particle2.velocity)]
                dx_dv = sum(dx*dv     for (dx,dv)     in zip (dx,dv))
                dx_2  = sum(dx_i*dx_i for dx_i        in dx)
                dv_2  = sum(dv_i*dv_i for dv_i        in dv)
                #assert dx_2 - 4*R**2>=0, f'Improper configuration t={t_simulated}, i={particle_index}, j={other}, anomaly={-dx_2 + 4*R**2}'
                disc  = dx_dv**2 - dv_2 * (dx_2 - 4*R**2)
     
                if disc>=0 and dx_dv<0:
                    return -( dx_dv +math.sqrt(disc))/dv_2   # FIXME            
            dt = get_next_collision(configuration[particle_index],configuration[other])
            if dt !=None:
                collision = configuration[particle_index].events[other]
                collision.t_expected = t_simulated + dt
                return collision
        
        return [
            collision for collision in [get_collision_with(other) for other in range(particle_index+1,len(configuration))]
            if collision!=None
        ]
  
    # act
    #
    # Reverse motion along normal at point of collision
    
    def act(self,configuration,topology): #Krauth Algorithm 2.3
        super().act(configuration,topology)
        k              = configuration[self.particle_index]
        l              = configuration[self.other_particle]
        delta_x        = [(pos_k -pos_l)         for (pos_k,pos_l) in zip(k.position,l.position)]
        delta_x_norm   = math.sqrt(sum(delta**2  for delta in delta_x ))
        e_perp         = [dx/delta_x_norm        for dx in delta_x]
        delta_v        = [(vel_k-vel_l)          for (vel_k,vel_l) in zip(k.velocity,l.velocity)]
        delta_v_e_perp = sum([v*e                for (v,e) in zip(delta_v,e_perp)])
        k.velocity     = [v - e* delta_v_e_perp  for (v,e)in zip(k.velocity,e_perp) ] 
        l.velocity     = [v +  e* delta_v_e_perp for (v,e)in zip(l.velocity,e_perp)]
        if delta_x_norm<2*topology.R:
            diff         = 2 * topology.R -delta_x_norm
            delta        = 0.5*diff + np.finfo(float).eps
            k.position   = [p + e*delta for (p,e) in zip(k.position,e_perp)]
            l.position   = [p - e*delta for (p,e) in zip(l.position,e_perp)] 
            delta_x      = [(pos_k -pos_l) for (pos_k,pos_l) in zip(k.position,l.position)]
            delta_x_norm = math.sqrt(sum(delta**2 for delta in delta_x ))            
        #assert delta_x_norm>=2*R
        return 1 # So we can count this collision

    # List of particles that are involved in this event.
    # Used to cull them so their next collision can be calculated.    
    def get_colliders(self):
        return [self.particle_index,self.other_particle]
    


# get_eta
#
# Density of particles
def get_eta(N,R,L=[1,1,1],D=3):
    return (N*(4/D)*math.pi*R**D)/(L[0]*L[1]*L[2])

def get_R(N,eta,L=[1,1,1],D=3):
    return ((L[0]*L[1]*L[2]*eta*D)/(4*N*math.pi))**(1/3)

    
# create_configuration
#
# Build particles for box particles
# Make sure that spheres don't overlap 
def create_configuration(N=100,R=0.0625,NT=25,E=1,L=[1,1,1]):
    def get_position():
        return [random.uniform(R-l,l-R) for l in L]
    
    # get_velocity
    def get_velocity(): #Krauth Algorithm 1.21
        velocities = [random.gauss(0, 1) for _ in L]
        sigma      = math.sqrt(sum([v**2 for v in velocities]))
        upsilon    = random.random()**(1/len(L))
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
            print (f'Radius = {R}, Density = {get_eta(N,R,L)}, {i+1} attempts')
            return i,product
        
        product= [Particle(position=get_position(),radius=R) for _ in range(N)]
        
    raise MolecularDynamicsError(f'Failed to create valid configuration for R={R}, density={get_eta(N,R,L)}, within {NT} attempts')

# Build dictionaries of all possible events. We will extract active events as we compute collisions
def link_events(configuration):
    
    for i in range(len(configuration)):
        for wall in Wall:
            configuration[i].events[wall]=HitsWall(i,wall)
        for j in range(i+1,len(configuration)):
            configuration[i].events[j]= Collision(i,j)
 
# flatten
#
# Flatten a list of lists into a simple list

def flatten(lists):
    return [item for sublist in lists for item in sublist]

# get_unaffected
#
# Get list of events that weren't affected by collisions
# i.e. particles that were not involved in collision

def  get_unaffected(events,particles_involved):
    # intersects
    #
    # Check to see whether two lists have an element in common
    def intersects(list1,list2):
        for i in list1:
            for j in list2:
                if i==j: return True
        return False
    
    return [event for event in events if not intersects(event.get_colliders(),particles_involved)]

# merge
#
# Merge two sorted lists into a single sorted list
def merge(events1,events2):
    events = []
    i      = 0
    j      = 0
    while i<len(events1) and j<len(events2):
        if events1[i]<events2[j]:
            events.append(events1[i])
            i+=1
        else:
            events.append(events2[j])
            j+=1 
            
    # Copy left over events
    while i<len(events1):
        events.append(events1[i])
        i+=1      
        
    while j<len(events2):
        events.append(events2[j])
        j+=1   
        
    return events

if __name__ == '__main__':
    import argparse,sys,matplotlib.pyplot as plt,time,pickle,os
    from scipy.stats import chisquare
    from matplotlib import rc
    from shutil import copyfile
    
    # create_parser
    #
    # Create parser for command line arguments
    
    def create_parser():
        default_plot_file = os.path.basename(__file__).split('.')[0]
        product = argparse.ArgumentParser('Molecular Dynamice simulation')
        product.add_argument('--N',    type=int,   default=25,                help='Number of particles')
        product.add_argument('--T',    type=float, default=100,               help='Maximum Time')
        product.add_argument('--R',    type=float, default=0.0625,            help='Radius of spheres')
        product.add_argument('--eta',  type=float, default=-1,               help='Density of spheres (alternative to R)')
        product.add_argument('--NT',   type=int,   default=100,               help='Number of attempts to choose initial configuration')
        product.add_argument('--NC',   type=int,   default=0,                 help='Minimum number of collisions')
        product.add_argument('--E',    type=float, default=1,                 help='Total energy')
        product.add_argument('--L',    type=float, default=1.0, nargs='+',    help='Half widths of box: one value or three.')
        product.add_argument('--seed', type=int,   default=None,              help='Seed for random number generator')
        product.add_argument('--freq', type=int,   default=100,               help='Frequency: number of steps between progress reports')
        product.add_argument('--show',             default=False,             help='Show plots at end of run',  action='store_true')
        product.add_argument('--plots',            default=default_plot_file, help='Name of file to store plots')
        product.add_argument('--save',             default=None,              help='Save configuration at end of run')
        product.add_argument('--load',             default=None,              help='Load configuration from saved file')
        product.add_argument('--topology',         default='Box',             help='Choose between Torus or Box topology')
        return product     
    
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

    # create_topology
    #
    # Create topology from command line or saved configuration
    
    def create_topology(topology,L,R):
        if topology.lower()=='box':
            return Box(L,R)
        if topology.lower()=='torus':
            return Torus(L,R)
        raise MolecularDynamicsError(f'Invalid topology specified: {topology}')
        
    # plot_results
    #
    # Produce histogram for energies, and compare with Boltzmann distrbution
    
    def plot_results(configuration,collision_count,R=0.0625,L=[1,1,1],plots='plots',topology=None):
        
        def plot_energies(axis=None,N=0):
            energies  = [particle.get_energy() for particle in configuration]
            E         = sum(energies)
            kT        = (2/3)*E/N  # Average energy of particle is 1.5kT        
            n,bins,_  = axis.hist(energies, bins='fd', # Freedman Diaconis Estimator
                                     label='Simulation', facecolor='b', edgecolor='b',fill=True)
            n,bins   = consolidate_bins(n,bins)
            xs       = [0.5*(a+b) for a,b in zip(bins[:-1],bins[1:])]   # mid points of bins
            ys       = [boltzmann(E,kT=kT) for E in xs]          
            scale_ys = sum(n)/sum(ys)   # We want area under Boltzmann to match area under energies
            y_scaled = [y*scale_ys for y in ys]
            chisq,p  = chisquare(n,y_scaled) # Null hypothesis: data has Boltzmann distribution
            axis.plot(xs, y_scaled, color='r', label='Boltzmann')
            axis.set_xlabel('Energy')
            axis.set_title(f'Energies: $\\chi^2$={chisq:.2f}, p={p:.3f}')
            axis.legend() 
            
        def plot_positions(axis=None,N=0):
            def get_chi_sq(n,bins,i):
                bin_widths = [b-a for (a,b) in zip(bins[:-1],bins[1:])]
                uniform = [N*w/2*L[i] for w in bin_widths]
                return chisquare(n[i],uniform)
        
            def get_title(n,bins):
                chisq_s= [get_chi_sq(n,bins,i) for i in range(len(n))]       
                chs    = ', '.join([f'{chi:.2f}' for (chi,_) in chisq_s])
                ps     = ', '.join([f'{p:.2f}' for (_,p) in chisq_s])
                return f'Positions $\\chi^2$=({chs}), p=({ps})'
            
            n,bins,_ = axis.hist([[particle.position[i] for particle in configuration] for i in range(len(L))],
                         label=['North-South','East-West','Top-Bottom'])    
            ys       = [N/(len(bins)-1) for _ in bins]
            axis.plot(bins, ys, '--', linewidth=1,color='r', label='Uniform')
            axis.legend(loc='best')  
            axis.set_title(get_title(n,bins))
            
        fig, axes = plt.subplots(2, 1, constrained_layout=True)
        N         = len(configuration)
        fig.suptitle(f'{topology.pretty()}: N={N}, $\\eta$={get_eta(N,R,L):.4f}, collisions={collision_count:,}')     
        plot_energies(axis=axes[0],N=N)
        plot_positions(axis=axes[1],N=N)
        fig.savefig(f'{plots}.png')
        
    # load_file
    #
    # Load a configuration that has been saved previously
    
    def load_file(file_name):
        try:  # try new file format first
            with open(file_name,'rb') as f:
                R,L,N,collision_count,topology_name,configuration = pickle.load(f)
                print (f'Restarting from {file_name}. R={R}, L={L}, N={len(configuration)}, topology={topology_name}')
                return R,L,N,collision_count,create_topology(topology_name,L,R),configuration
        except (ValueError,EOFError) : # otherwise fall back on old configuration
            with open(file_name,'rb') as f:
                R,L,N,collision_count,configuration = pickle.load(f)
                print (f'Restarting from {file_name}. R={R}, L={L}, N={len(configuration)}')
                return R,L,N,collision_count, create_topology('box',L,R),configuration
    
    # save_file
    #
    # Save a configuration so it can be restarted later
    #
    # Also backup previously saved file
    def save_file(file_name):        
        if file_name!=None:
            if os.path.exists(file_name):
                copyfile(file_name,'~'+file_name)
            with open(file_name,'wb') as save_file:
                pickle.dump((R,L,N,collision_count,topology.name(),configuration),save_file)

    # consolidate_bins
    #
    # Consolidate bins so none (except possible the last) has fewer items than a specified quota
    
    def consolidate_bins(n,bins,quota=5):
        new_counts        = []
        consolidated_bins = [bins[0]]
        carried_over      = 0
        
        for i in range(len(n)):
            carried_over+=n[i]
            if carried_over>=quota:
                new_counts.append(carried_over)
                consolidated_bins.append(bins[i+1])
                carried_over = 0
                
        if carried_over>=quota:
            new_counts.append(carried_over)
            consolidated_bins.append(bins[-1])
            
        return (new_counts,consolidated_bins)
    
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)      
    
    args            = create_parser().parse_args()   
    L               = get_L(args.L)
    N               = args.N
    R               = args.R if args.eta == -1 else get_R(N,args.eta)
      
    collision_count = 0   # Number of particle-particle collisions - walls not counted
    random.seed(args.seed)
    topology = create_topology(args.topology,L,R)

    try:
        start_elapsed_time = time.time()
        configuration = None
 
        if args.load==None:
            _,configuration = create_configuration(N=N, R=R, NT=args.NT, E=args.E, L=L )
        else: # restart from saved configuration
            R,L,N,collision_count,topology,configuration = load_file(args.load)               
                
        link_events(configuration)
        
        init_elapsed_time = time.time()
        print (f'Time to initialize: {(init_elapsed_time-start_elapsed_time):.1f} seconds')
  
        
        # Build a sorted list of events. After each collision we will remove all events for
        # particles involved in the collision, and:
        # 1. Evolve configuration
        # 2. Generate new events from particles involved in collission, and merge them with list of events.
        
        hits_wall         = [HitsWall.get_collisions(configuration[i],t_simulated=0,L=L,R=R) for i in range(N)] 
        particle_particle = [Collision.get_collisions(i,configuration,t_simulated=0,R=R) for i in range(N)]
        events            = sorted(flatten(hits_wall + particle_particle))
        t_simulated       = 0
        step_counter      = 0   # Used to decide whether to print progress indicator for a particular iteration
        
        # This is the simulation, during which time moves forward
        #      A.   Simulated time
        #      B.   Particle moves depending on velocity and time step

        while t_simulated< args.T or collision_count < args.NC:
            event         = events[0]   # Get first event 
            dt            = event.t_expected - t_simulated  # Duration until event is expected
            
            #      A.   Update simulated time to time when event occurs
            t_simulated   = event.t_expected
            
            # Update step counter and see whether it is time to print
            
            step_counter += 1
            if step_counter%args.freq==0:
                print (f'{event}, collisions={collision_count}')  
                
            #      B.   Move all particles to their estimated positions at time event occurs
            for particle in configuration:
                particle.evolve(dt)
             
            # Perform event and update collision count   
            collision_count   += event.act(configuration,topology)
            
            # Find out which particles were involved - one for a wall collition, 2 for particle-particles
            particles_involved = event.get_colliders()
            
            # Remove those who were involved
            events_retained    = get_unaffected(events,particles_involved)
                 
            # Recalculate events from  particles that were involved: no others are affected!
            # Note that we end up with lists of lists of events, so will need to flatten
            hits_wall         = [HitsWall.get_collisions(configuration[i],
                                                         t_simulated=t_simulated,
                                                         L=L,
                                                         R=R) for i in particles_involved]
            particle_particle = [Collision.get_collisions(i,
                                                          configuration,
                                                          t_simulated=t_simulated,
                                                          R=R) for i in particles_involved]
            
            new_events        = flatten(particle_particle + hits_wall)
            
            events            =  merge(events_retained, sorted(new_events))        

  
                
        # Simulation is over: compute elapsed (computer) time
        end_elapsed_time       = time.time()
        collision_elapsed_time = end_elapsed_time - init_elapsed_time
        total_elapsed_time     = end_elapsed_time - start_elapsed_time
        print (f'Time to simulate {collision_count} collisions between {N} particles: {collision_elapsed_time:.1f} seconds') 
        print (f'Total Time: {total_elapsed_time:.1f} seconds')
        
        save_file(args.save)
        
        plot_results(configuration,collision_count,R=R,L=L,plots=args.plots,topology=topology)

        if args.show:
            plt.show()
            
    except MolecularDynamicsError as e:
        print (e)
        sys.exit(1)
