''' Python PSO implementations.
    This new_version is to return the cluster Node
    with the maximum execution time as the new makspan time '''
import random
import math
import copy
import sys
import numpy
import time
from numpy.random import normal
from numpy import rint
import collections
##----------Generating Data-----------------------------------------------------
''' To generate tasks execution times data according to clusters observation '''
def prMatrix(x):
    for row in x:
        for val in row:
            print(val,end=',')
        print()
    print()
n=30 ## Thenumber of tasks to be allocated
''' Estimated extecution time tasks take on nodes.
    50 means Pi2B, 40 means Pi3B, 10 means Pi4B '''
x = [
    [50]*n, [50]*n, [50]*n,             ## 3nodes (RPi2B)
    [40]*n, [40]*n, [40]*n, [40]*n,     ## 4nodes (RPi3B
    [10]*n                              ## 1node  (RPi4B)
    ]
print('Initial tasks execution times:')
prMatrix(x)
''' Code to randomize data created by Jeremy '''
####### @jsinger new perturbation code for cost matrix
####### thresholds for changing costs
COEFFICIENT_OF_VARIATION=0.5  ## (c.o.v. = stdev / mean = sigma/mu)
for i in range(len(x)):
    for j in range(len(x[i])):
        mu = x[i][j]
        sigma = COEFFICIENT_OF_VARIATION * mu
        updated_value = int(rint(normal(mu, sigma)))
        x[i][j] = max(0, updated_value)
print('Estimated tasks execution time:')
prMatrix(x)
costs = x  #execution time cost
##------Nodes Capacities Function-----------------------------------------------
def nodes_cap(particle):
    ''' Check if a PSO particle meets the nodes capacity constraints.
        return True if it meets the constraints.
        return False otherwise '''
    node_cap = [n*0.15]*3 + [n*0.2]*4 + [n*0.6]*1 ## 3nodes(RPi2), 4nodes(RPi3), 1node(RPi4)
    particle = [int (i) for i in particle]
    counter=collections.Counter(particle)
    for i in (counter):
        if (counter.get(i) > node_cap[i]):
            return False
    return True
##------Fitness Function--------------------------------------------------------
''' Old fintess function (The sum of tasks execution times)'''
def fitness_fun(particle):
    num_nodes = len(costs)
    num_tasks = len(particle)
    fitnessVal = 0.0
    ''' Call nodes_cap function to check node capacity constraints
        if nodes constraints is False, FitnessVal is (inf).
        if nodes constraints is True,  FirnessVal is calculated '''
    if (nodes_cap(particle) == False):
        fitnessVal = float('inf')
    else:
        for t in range(num_tasks):
            node = int (particle[t])
            assert node <= num_nodes
            fitnessVal += costs [node][t]
    return fitnessVal
##------Function to return the devices with maximum exctime---------------------
def findmax(best_position):
    '''To Return the node with maximum execution time for old fitness function'''
    best_position = [int (i) for i in best_position]
    print ("Best Nodes Allocation:", best_position) ##List 1 (allocation nodes)
    fitnessVal = 0.0
    excTime = []
    for task in range(len(best_position)):
        node = int (best_position[task])
        excTime.append(costs [node][task])
    print ("Tasks excTime:", excTime) ## List 2 (exctimes)
    sums = {}
    for key, value in zip(best_position,excTime):
        try:
            sums[key] += value
        except KeyError:
            sums[key] = value
    print("Nodes Execution Time: ", sums)
    maximum = max(sums, key=sums.get)
    print("Node:",maximum, "has MaxExecutionTime:",sums[maximum], "sec")
    return sums[maximum]

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
makespanVal = findmax(best_position)
print("Makspan of best solution = %.6f" % makespanVal)
print("\nEnd Particle Swarm Optimization\n")
etime = time.time()
print ("Time =", etime-stime)
