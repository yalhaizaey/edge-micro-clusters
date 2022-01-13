# python implementation of particle swarm optimization (PSO)
import random
import math    # cos() for Rastrigin
import copy    # array-copying convenience
import sys     # max float
import numpy
import time
from numpy.random import normal
from numpy import rint


##----------Generating Data--------------
##randomize data;
def prMatrix(x):
    for row in x:
        for val in row:
            print(val,end=',')
        print()
    print()

# example array of task costs on different nodes
n= 100 #number of tasks to be allocated

# estimated extecution time tasks take on nodes. 50 means Pi2B, 40 means Pi3B, 10 means Pi4B
x = [
    [50]*n, [50]*n, [50]*n, [50]*n, [50]*n, [50]*n, [50]*n, [50]*n, [50]*n,

    [40]*n, [40]*n, [40]*n, [40]*n, [40]*n, [40]*n, [40]*n, [40]*n, [40]*n,

    [10]*n, [10]*n,
    ]
print (x)
print('initial x:')
print('----------')
prMatrix(x)
####### @jsinger new perturbation code for cost matrix
# thresholds for changing costs
COEFFICIENT_OF_VARIATION=0.5  # c.o.v. = stdev / mean = sigma/mu
# try different values - between 0 and 1?
for i in range(len(x)):
    for j in range(len(x[i])):
        mu = x[i][j]
        sigma = COEFFICIENT_OF_VARIATION * mu
        updated_value = int(rint(normal(mu, sigma)))
        x[i][j] = max(0, updated_value)  # no negative costs!

print('final x:')
print('----------')
prMatrix(x)

#execution time cost
costs = x

#-------fitness functions----------------
def fitness_fun(position):
    particle =(position) # particle = solution
    num_nodes = len(costs)
    num_tasks = len(particle)
    fitnessVal = 0.0
    for t in range(num_tasks):
        node = int (particle[t])
        assert node <= num_nodes
        fitnessVal += costs [node][t]
    return fitnessVal

#-------------------------#-------------------------
#particle class
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
    self.fitness = fitness(self.position) # curr fitness

    # initialize best position and fitness of this particle
    self.best_part_pos = copy.copy(self.position)
    self.best_part_fitnessVal = self.fitness # best fitness

#-------------------------#-------------------------
# particle swarm optimization function
def pso(fitness, max_iter, n, dim, minx, maxx):
  # hyper parameters
  w = 0.729    # inertia
  c1 = 1.49445 # cognitive (particle)
  c2 = 1.49445 # social (swarm)
  rnd = random.Random()

  # create n random particles
  swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = [0.0 for i in range(dim)]
  best_swarm_fitnessVal = sys.float_info.max # swarm best

  # computer best particle of swarm and it's fitness
  for i in range(n): # check each particle
    if swarm[i].fitness < best_swarm_fitnessVal:
      best_swarm_fitnessVal = swarm[i].fitness
      best_swarm_pos = copy.copy(swarm[i].position)

  # main loop of pso
  Iter = 0
  while Iter < max_iter:

    # after every 10 iterations
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)

    for i in range(n): # process each particle

      # compute new velocity of curr particle
      for k in range(dim):
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()

        swarm[i].velocity[k] = (
                                 (w * swarm[i].velocity[k]) +
                                 (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                                 (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                               )

      # compute fitness of new position
      swarm[i].fitness = fitness(swarm[i].position)

      # is new position a new best for the particle?
      if swarm[i].fitness < swarm[i].best_part_fitnessVal:
        swarm[i].best_part_fitnessVal = swarm[i].fitness
        swarm[i].best_part_pos = copy.copy(swarm[i].position)

      # is new position a new best overall?
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

    # for-each particle
    Iter += 1
  #end_while
  return best_swarm_pos
# end pso

#----------------------------#-------------------------
# Driver code
import time
stime = time.time()
print("\nBegin Particle Swarm Optimization \n")
dim = n  #here is the number of tasks (n is the number of tasks)
fitness = fitness_fun

print("Goal is to minimize cost function in " + str(dim) + " variables")

for i in range(dim-1):
    print("0, ", end="")
print("0")

num_particles = 50
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting PSO algorithm\n")


best_position = pso(fitness, max_iter, num_particles, dim, 0, 19)

print("\nPSO completed\n")
print("\nBest solution found:\n")
print(["%.6f"%best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("\nfitness of best solution = %.6f" % fitnessVal)

print("\nEnd Particle Swarm Optimization\n")
etime = time.time()
print ("time =", etime-stime)
