''' Python PSO implementations. Linear Model.'''
import random
import math
import copy
import sys
import numpy
import time
from numpy.random import normal
from numpy import rint
import collections
##------------------------------------------------------------------------------
n=100 ## number of tasks
## new_costs = Yolo Data, PS Data, Aeneas Data
new_costs = [
            [{"m":10, "c":45}, {"m":10, "c":25}, {"m":7, "c":7}],   # pi2 (0)
            [{"m":10, "c":45}, {"m":10, "c":25}, {"m":7, "c":7}],   # pi2 (0)
            [{"m":10, "c":45}, {"m":10, "c":25}, {"m":7, "c":7}],   # pi2 (0)

            [{"m":15, "c":20}, {"m":10, "c":10}, {"m":5, "c":3}],   # pi3 (1)
            [{"m":15, "c":20}, {"m":10, "c":10}, {"m":5, "c":3}],   # pi3 (1)
            [{"m":15, "c":20}, {"m":10, "c":10}, {"m":5, "c":3}],   # pi3 (1)
            [{"m":15, "c":20}, {"m":10, "c":10}, {"m":5, "c":3}],   # pi3 (1)

            [{"m":5, "c":5}  , {"m":5, "c":7}, {"m":2, "c":5}]    # pi4 (2)
            ];
##------Nodes Capacities Function-----------------------------------------------
def nodes_cap(particle):
    ''' check if a particle meets the nodes capacity constraints.
        return True if meet the constraints.
        return False otherwise '''
    node_cap = [n*0.15]*3 + [n*0.2]*4 + [n*0.6]*1 ## 3nodes(RPi2), 4nodes(RPi3), 1node(RPi4)
    particle = [int (i) for i in particle]
    counter=collections.Counter(particle)
    for i in (counter):
        if (counter.get(i) > node_cap[i]):
            return False
    return True
#-------New Fitness Function (y=mx+b)-------------------------------------------
''' New fitness function based on linear observations (y=mx+b) '''
def fitness_fun(particle):
    ''' particle is a list - n integers - one integer per task,
        the integer determines which _node_ this indexed task will be allocated'''
    particle = [int (i) for i in particle]
    fitnessVal = 0

    ''' counter to calculate how many tasks assinged to each node in a particle'''
    counter = collections.Counter(particle)
    for node in counter:
        numTasks = counter.get(node)
        #print ("on node ", node, " there are ", numTasks, " tasks");
    for node in counter:
        # find out how many tasks are running on that node
        numTasks = counter.get(node)
        # use the m and c values to determine the
        m = new_costs[node][0]['m'] # are all tasks the same?
        c = new_costs[node][0]['c']

        ''' call nodes_cap function to check node capacity constraints
            if nodes constraints is False, FitnessVal is (inf)
            if nodes constraints is True,  FirnessVal is calculated '''
        if (nodes_cap(particle) == False):
            fitnessVal = float('inf')
        else:
            time = m*numTasks + c ## new fitness based on linear equation
            fitnessVal += time
    return fitnessVal
##------Start PSO---------------------------------------------------------------
''' Staring PSO Optimisation'''
class Particle:
  def __init__(self, fitness, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)

    # initialize position of the particle with 0.0 value
    self.position = [0.0 for i in range(dim)]

    # initialize velocity of the particle with 0.0 value
    self.velocity = [0.0 for i in range(dim)]

    # initialize best particle position of the particle with 0.0 value
    self.best_part_pos = [0.0 for i in range(dim)]

    # loop dim times to calculate random position and velocity
    # range of position and velocity is [minx, max]
    for i in range(dim):
      self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
      self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)

    # compute fitness of particle
    self.fitness = fitness(self.position)

    # initialize best position and fitness of this particle
    self.best_part_pos = copy.copy(self.position)
    self.best_part_fitnessVal = self.fitness

def pso(fitness, max_iter, n, dim, minx, maxx):
  ''' Defineing PSO Hyperparameters '''
  w = 0.729    # inertia wieght
  c1 = 1.49445 # cognitive (particle)
  c2 = 1.49445 # social (swarm)
  rnd = random.Random()

  # create n random particles
  swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = [0.0 for i in range(dim)]
  best_swarm_fitnessVal = sys.float_info.max # swarm best

  # computer best particle of swarm and it's fitness
  for i in range(n):
    if swarm[i].fitness < best_swarm_fitnessVal:
      best_swarm_fitnessVal = swarm[i].fitness
      best_swarm_pos = copy.copy(swarm[i].position)

  Iter = 0
  while Iter < max_iter:
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)

    for i in range(n):
      # compute new velocity of curr particle
      for k in range(dim):
        r1 = rnd.random()
        r2 = rnd.random()
        swarm[i].velocity[k] = (
                               (w * swarm[i].velocity[k]) +
                               (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                               (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                               )
      ''' compute fitness of new position '''
      swarm[i].fitness = fitness(swarm[i].position)

      # Is new position a new best for the particle?
      if swarm[i].fitness < swarm[i].best_part_fitnessVal:
        swarm[i].best_part_fitnessVal = swarm[i].fitness
        swarm[i].best_part_pos = copy.copy(swarm[i].position)

      # Is new position a new best overall?
      if swarm[i].fitness < best_swarm_fitnessVal:
        best_swarm_fitnessVal = swarm[i].fitness
        best_swarm_pos = copy.copy(swarm[i].position)

      # Added by Yousef! -- compute new position using new velocity & clip new position between minx and maxx
      for k in range(dim):
        swarm[i].position[k] += swarm[i].velocity[k]
        if swarm[i].position[k] < minx:
            swarm[i].position[k] = minx
        elif swarm[i].position[k] > maxx:
            swarm[i].position[k] = maxx
    Iter += 1
  return best_swarm_pos
##----------------End-of-PSO ---------------------------------------------------
''' Driver Code '''
stime = time.time()
print("\nBegin Particle Swarm Optimization \n")
dim = n  #here is the number of tasks (n is the number of tasks)
fitness = fitness_fun
num_particles = 50
max_iter = 100

print("Goal is to minimize cost function in " + str(dim) + " variables")
print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, 0, 7)

print("\nPSO completed\n")
print("\nBest solution found:")
best_position = [int (i) for i in (best_position)]
print (best_position)

fitnessVal = fitness(best_position)
print("\nFitness of best solution = %.6f" % fitnessVal)
print("\nEnd Particle Swarm Optimization\n")
etime = time.time()
print ("Time =", etime-stime)
