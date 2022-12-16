#######################################################
##### PARTICLE SWARM OPTIMIZATION (PSO) ALGORITHM #####
############################### -Siddharth Yadav- #####

####======== main.py ========####
# Contains the source code for the PSO algorithm

# Importing the necessary libraries
import numpy as np
import math
from functools import reduce
from copy import deepcopy
import matplotlib.pyplot as plt
from joblib import dump
from plot import plot_pso


# Defining a particle class to simulate a particle
class Particle:
    # Initializes the particles with appropriate attributes
    def __init__(self, inertia_component, cognitive_component, social_component, fitness_function, minimize=True):
        self.inertia_component = np.round(
            1-inertia_component, decimals=2)  # Resistance to movement
        self.cognitive_component = cognitive_component  # Influence of personal best
        self.social_component = social_component  # Influence of global/swarm best
        self.fitness_function = fitness_function  # The function being optimized
        self.minimize = minimize  # Whether the optimization is minimizing or not

        # Initializes random x,y coordinates in a domain space
        self.position = np.random.randint(-15, 16, 2)
        # Personal best position (initially, will be the starting position)
        self.bestposition = self.position
        self.bestfitness = self.fitness()  # Fitness of the personal best solution
        # Initializes a random direction vector for the particle
        self.direction = np.random.randn(2)

    # Returns the fitness for the current position
    def fitness(self):
        x, y = self.position
        return self.fitness_function(x, y)

    # Updates the best position if the current position is better
    def update_best_position(self):
        if self.minimize:  # For minimizing the fitness
            if self.fitness() < self.bestfitness:
                self.bestfitness = self.fitness()
                self.bestposition = self.position
        else:  # For maximizing the fitness
            if self.fitness() > self.bestfitness:
                self.bestfitness = self.fitness()
                self.bestposition = self.position

    # Updates the best position depending on environmental factors
    def update_position(self, swarm_best_position):
        cog_vector = (self.bestposition - self.position) * \
            self.cognitive_component * \
            np.random.random()  # Vector towards the personal best (scaled by cognitive component)
        social_vector = (swarm_best_position - self.position) * \
            self.social_component * \
            np.random.random()  # Vector towards the global/swarm best (scaled by social component)
        new_position = self.position + \
            (self.direction*self.inertia_component) + cog_vector + \
            social_vector  # New position is influenced by three vectors: cognitive, social and current direction (scaled by inertia_component)

        self.direction = new_position - self.position # Recalculates the direction vector based the new position
        self.position = new_position  # Sets the new position as the current position
        self.update_best_position()  # Updates the best position

    # Prints a report of the particle's coordinates and fitness
    def print_report(self):
        print(
            f"\tx:{self.position[0]:+.3f}\ty:{self.position[1]:+.3f}\tfitness:{self.fitness():+.3f}")


# Defining a swarm class to simulate a swarm (group of particles)
class Swarm:
    # Intializes the swarm with the appropriate attributes
    def __init__(self, num_particles, iterations, inertia_component, cognitive_component, social_component, fitness_function, minimize=True, verbose=False, dump_swarm=False):
        self.num_particles = num_particles  # The total number of particles in the swarm
        # Number of steps/iterations to be perfomed by the particles
        self.iterations = iterations
        self.inertia_component = np.round(1-inertia_component, decimals=2)
        self.cognitive_component = cognitive_component
        self.social_component = social_component
        self.fitness_function = fitness_function
        self.minimize = minimize
        self.verbose = verbose  # Sets the frequency of logging outputs and reports
        
        # Generates a collection of swarm particle positions to enable external ploting and analysis
        self.dump_swarm = dump_swarm
        if self.dump_swarm:
            self.collection = []  # The collection of positions if 'dump_swarm' was set to True

    # Generates the prescribed number of swarm particles
    def initialize_swarm(self):
        self.swarm = [Particle(self.inertia_component, self.cognitive_component, self.social_component,
                               self.fitness_function, self.minimize) for _ in range(self.num_particles)]

        # Sets the best particle from the initial swarm using the reduce function
        if self.minimize:
            self.best_particle = reduce(lambda x,y: x if x.fitness() < y.fitness() else y, self.swarm)
        else:
            self.best_particle = reduce(lambda x,y: x if x.fitness() > y.fitness() else y, self.swarm)

        if self.verbose: # Print the initial swarm if 'verbose' is True
            print(f"{'Initial swarm':*^60}")
            self.swarm_report()

        if self.dump_swarm: # Save the positions of the initial swarm to collection if 'dump_swarm' is True
            self.dump_positions()

    # Updates the best particle of the swarm if a particle from current group is better
    # (Note) deepcopy is required to save current state of the best particle and avoid side-effects from shared references 
    def update_swarm_best_particle(self):
        if self.minimize:
            for particle in self.swarm:
                if particle.fitness() < self.best_particle.fitness():
                    self.best_particle = deepcopy(particle)
        else:
            for particle in self.swarm:
                if particle.fitness() > self.best_particle.fitness():
                    self.best_particle = deepcopy(particle)
    
    # Updates the position of each particle in the swarm while providing it the knowledge of global best solution
    def update_swarm(self):
        for particle in self.swarm:
            particle.update_position(self.best_particle.position)
    
    # Runs the Particle-Swarm-Optimization (PSO) algorithm for the defined number of iterations
    def run_pso(self):
        for iteration in range(self.iterations):
            self.update_swarm()
            self.update_swarm_best_particle()

            if self.verbose: # Print swarm report per iteration if 'verbose' is True
                print_str = f"Iteration: {iteration+1}"
                print(f"\n{print_str:*^60}")
                self.swarm_report()

            if self.dump_swarm: # Save the positions of the current swarm to collection if 'dump_swarm' is True
                self.dump_positions()
                
            print("\nGlobal best solution:") # Priting the global best solution after every iteration
            self.best_particle.print_report()

    # Print the report for each particle in the current swarm and highlighting its best
    def swarm_report(self):
        if self.minimize:
            best_fitness = min(particle.fitness() for particle in self.swarm)
        else:
            best_fitness = max(particle.fitness() for particle in self.swarm)
        
        for particle in self.swarm:
            if particle.fitness() == best_fitness:
                print(
                    f"(best)  x:{particle.position[0]:+.3f}\ty:{particle.position[1]:+.3f}\tfitness:{particle.fitness():+.3f}")
            else:
                particle.print_report()

    # Save the position of each particle in swarm to collections
    def dump_positions(self):
        self.collection.append([particle.position for particle in self.swarm])


####======== FITNESS FUNCTIONS ========####
# Listing of several fitness functions to optimize
# (Note) The global optimum and domain values for each function is provided to help assess the success of the algorithm
# and provided rough guidelines during the plotting/animation phase

# Booth's function (single global minimum with value 0: x=1, y=3 | domain space: -10<= x,y <=10)
def booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2


# Bukin's function N.6 (single global minimum with value 0: x=-10, y=1 | domain space: -15<= x <=-5, -3<= y <=3)
def bukin_No_6(x, y):
    return 100*math.sqrt(np.fabs(y-0.01*x**2))+0.01*np.fabs(x+10)


# Himmelblau's function
# (four global minima with value 0: x=3,y=2| x=-2.805118,y=3.131312| x=-3.779310,y=-3.283186| x=3.584428,y=-1848126
# domain space: -5<= x,y <=5)
def himmelblau(x, y):
    return (x**2+y-11)**2 + (x+y**2-7)**2


# Goldstein-Price function (single global minimum with value 3: x=0, y=-1 | domain space: -2<= x,y <=2)
def goldstein_price(x, y):
    a = (x+y+1)**2
    b = (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    c = (2*x - 3*y)**2
    d = (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return (1 + a * b) * (30 + c * d)


####======== GLOBAL PARAMETERS FOR THE PSO ALGORITHM ========####
ITERATIONS = 200
NUM_PARTICLES = 200
INERTIA_COMPONENT = 0.8
COGNITIVE_COMPONENT = 0.8
SOCIAL_COMPONENT = 0.05
FITNESS_FUNCTION = bukin_No_6


####======== RUNNING THE PSO ALGORITHM ========####
# Creating the swarm object
swarm = Swarm(NUM_PARTICLES, ITERATIONS, INERTIA_COMPONENT, COGNITIVE_COMPONENT,
              SOCIAL_COMPONENT, FITNESS_FUNCTION, minimize=True, verbose=False, dump_swarm=True)

# Initializing the swarm
swarm.initialize_swarm()

# Running the pso algorithm
swarm.run_pso()

# Saving the position of particles collected over every iteration to a local file for external plotting and analysis
dump(swarm.collection, 'swarm_collection.joblib') # All positions
dump(swarm.best_particle.position, 'best_position.joblib') # Position of the global best solution discovered

# Providing the function name and ploting the results
function_name = ' '.join(x.capitalize() for x in FITNESS_FUNCTION.__name__.split('_'))

# plot_pso(function_name=function_name)
