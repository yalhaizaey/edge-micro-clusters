''' Python PSO implementations. Linear Model.'''
import copy
import collections
import math
import numpy
from numpy.random import normal
from numpy import rint
import random
import sys
import time

##-------------To generate/estimate tasks type----------------------------------
''' To define number of tasks to be allocated and tasks types
    n is the total number of tasks.
    Max_Task_id is to define how many types of tasks (e.g., we have 3 applications)'''
n=45 ## define number of tasks to be allocated
Max_Task_id=2 ## define number of applications

'''Function to generate random task type'''
def randomTaskArray(n):
    taskArray = []
    for i in range (n):
         taskArray.append(random.randint(0,Max_Task_id))
    return (taskArray)
'''Function to generate uniform task type'''
def uniformTaskArray (n, taskId):
    return [taskId]*n
'''Function to generate equal task type'''
def equalPartsTaskArray(n):
    i=0
    tasks = list(range(0,Max_Task_id+1))
    taskArray=[]
    while i<n:
        taskArray.append(task[i%len(tasks)])
        i = i+1
    return taskArray

''' Task_type is an arry to append the tasks types.
    [0,0,0] means yolo, yolo, yolo '''


#task_type = uniformTaskArray(n,0) ## similer tasks
task_type = [0,1,2] * n ## Different tasks
print ("Task_Type:", task_type)
##-------------To define new cost matrix (y=mx+c)-------------------------------
''' To define the gradient (m) and the y-intercept (c)of different tasks on nodes '''
## the order of the new cost is as follow: [yolo, aeneas, ps]
new_costs = [
            [{"m":10, "c":50}, {"m":5, "c":15}, {"m":15, "c":20}],   #pi2
            [{"m":10, "c":50}, {"m":5, "c":15}, {"m":15, "c":20}],   #pi2
            [{"m":10, "c":50}, {"m":5, "c":15}, {"m":15, "c":20}],   #pi2

            [{"m":10, "c":30}, {"m":5, "c":5}, {"m":8, "c":15}],   #pi3
            [{"m":10, "c":30}, {"m":5, "c":5}, {"m":8, "c":15}],   #pi3
            [{"m":10, "c":30}, {"m":5, "c":5}, {"m":8, "c":15}],   #pi3

            [{"m":10, "c":25}, {"m":5, "c":5}, {"m":8, "c":15}],   #pi3B+

            [{"m":5, "c":5},   {"m":3, "c":5}, {"m":5, "c":5}],     #pi4
            ];
##------Nodes Capacities Function-----------------------------------------------
def nodes_cap(particle):
    ''' check if a particle meets the nodes capacity constraints.
        return True if meet the constraints.
        return False otherwise '''
    node_cap = [n*0.2]*3 + [n*0.2]*4 + [n*0.7]*1
    particle = [int (i) for i in particle]
    counter=collections.Counter(particle)
    for i in (counter):
        if (counter.get(i) > node_cap[i]):
            return False
    return True

#-------New fitness function based on the linear model (y=mx+b)-----------------
''' New fitness function based on linear observations (y=mx+b) '''
def fitness_fun(particle):
    ''' particle is a list - n integers - one integer per task,
        the integer determines which _node_ this indexed task will be allocated'''
    particle = [int (i) for i in particle]
    fitnessVal = 0
    ''' counter to calculate how many tasks assinged to each node in a particle'''
    counter = collections.Counter(particle)
    for node in counter:
        nodeFitness=0
        numTasks = counter.get(node)
        tasksonthisnode = [0]*(Max_Task_id+1)
        for i in range(len(particle)):
            if (particle[i] == node):
                taskId=task_type[i]
                tasksonthisnode[taskId]+=1
        ''' call nodes_cap function to check node capacity constraints
            if nodes constraints is False, FitnessVal is (inf)
            if nodes constraints is True,  FirnessVal is calculated '''
        if (nodes_cap(particle) == False):
            nodeFitness = float('inf')
        else:
            time = 0
            for taskId in range(0,Max_Task_id+1):
                m=new_costs[node][taskId]['m']
                if (tasksonthisnode[taskId]==0):
                    c=0;
                else:
                    c=new_costs[node][taskId]['c']
                ''' Fitness Function to evaluate the paritcles '''
                time += m*tasksonthisnode[taskId]+c
            nodeFitness=time
        if (nodeFitness>fitnessVal):
            fitnessVal=nodeFitness
    return fitnessVal

##----------------Function to map/interpret the solutions to nodes -------------
''' Function to map particles the the cluster nodes ip addresses '''
def map_cluster_nodes(particle):
    nodesip = numpy.array([]);
    for i in (particle):
        if (i==0):
            nodesip=numpy.append(nodesip,'192.168.1.229');
        elif (i==1):
            nodesip=numpy.append(nodesip,'192.168.1.224');
        elif (i==2):
            nodesip=numpy.append(nodesip,'192.168.1.222');
        elif (i==3):
            nodesip=numpy.append(nodesip,'192.168.1.230');
        elif (i==4):
            nodesip=numpy.append(nodesip,'192.168.1.148');
        elif (i==5):
            nodesip=numpy.append(nodesip,'192.168.1.168');
        elif (i==6):
            nodesip=numpy.append(nodesip,'192.168.1.182');
        elif (i==7):
            nodesip=numpy.append(nodesip,'192.168.1.131');
    nodesip1=[];
    nodesip1=(str(nodesip).strip('[]'))
    finalIPsBashFormat='('
    finalIPsBashFormat+=nodesip1
    finalIPsBashFormat+=')'
    ''' To wirte results and map ip address to nodes'''
    with open('pso_allocation.txt', 'w') as f:
        f.write(str(finalIPsBashFormat))

##------Start PSO Oprimisaition-------------------------------------------------
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
''' The main starts form here'''
stime = time.time()
print("\nBegin Particle Swarm Optimization \n")
dim = n  ##here is the number of tasks (n is the number of tasks)
cluster_nodes = len(new_costs) ## here is number of nodes in the cluster
print ("Number of Tasks:", n, "Cluster Nodes:", cluster_nodes)
fitness = fitness_fun
num_particles = 50
max_iter = 100

print("\nStarting PSO algorithm\n")

''' Start of PSO optimisation '''
best_position = pso(fitness, max_iter, num_particles, dim, 0, cluster_nodes-1)
print("\nPSO completed\n")

''' Convert best solution to integer '''
print("\nBest solution found:")
best_position = [int (i) for i in (best_position)]
print (best_position)

''' Call FitnessVal Function to compute the fitess of best solution '''
fitnessVal = fitness(best_position)
print("\nEstimated Makespan = %.6f" % fitnessVal)

''' Call map_results Function to mapp particle to cluster nodes ip address '''
map_cluster_nodes(best_position)
print("\nEnd Particle Swarm Optimization\n")
etime = time.time()
print ("AllocationTime=", etime-stime)

''' To write the estimated Makespan in file'''
with open('pso_est_makespan.txt', 'a') as f:
    f.write("%s\n" % str(fitness(best_position)))
