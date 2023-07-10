''' Python code to generate random allocation '''
import collections
import numpy
from numpy.random import normal
from numpy import rint
import random

''' defign numner of tasks and cluster size and number of applications'''
task_Num=100    ## the total number of tasks to be allocated
cluster_Size=7 ## cluster nodes
Max_Task_id=2  ## number of application

'''Function to generate uniform task type, this is becuase we run defog applications once at a time'''
def uniformTaskArray (task_Num, taskId):
    return [taskId]*task_Num
task_type = uniformTaskArray(task_Num,0)
#task_type = [0,1,2] * task_Num ## this is for different tasks
print ("Task_Type:", task_type)

''' Function to generate random allocation to cluster nodes'''
def randomAllocation(task_Num):
    randomArray = []
    for i in range (task_Num):
         randomArray.append(random.randint(0,cluster_Size))
    return (randomArray)
random_Allocation = randomAllocation(task_Num)
print ("random_Allocation:", random_Allocation)

''' To define the gradient (m) and the y-intercept (c)of different tasks on nodes '''
## yolo, aeneas, ps
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
''' New fitness function based on linear observations (y=mx+b) '''
def fitness_fun(random_Allocation):
    random_Allocation = [int (i) for i in random_Allocation]
    fitnessVal = 0
    ''' counter to calculate how many tasks assinged to each node in a particle'''
    counter = collections.Counter(random_Allocation)
    for node in counter:
        nodeFitness=0
        numTasks = counter.get(node)
        tasksonthisnode = [0]*(Max_Task_id+1)
        for i in range(len(random_Allocation)):
            if (random_Allocation[i] == node):
                taskId=task_type[i]
                tasksonthisnode[taskId]+=1
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

''' Function to map particles the the cluster nodes ip addresses '''
def map_cluster_Size(random_Allocation):
    nodesip = numpy.array([]);
    for i in (random_Allocation):
        if (i==0):
            nodesip=numpy.append(nodesip,'192.168.1.229');
        elif (i==1):
            nodesip=numpy.append(nodesip,'192.168.1.222');
        elif (i==2):
            nodesip=numpy.append(nodesip,'192.168.1.224');
        elif (i==3):
            nodesip=numpy.append(nodesip,'192.168.1.148');
        elif (i==4):
            nodesip=numpy.append(nodesip,'192.168.1.230');
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
    with open('rand_allocation.txt', 'w') as f:
        f.write(str(finalIPsBashFormat))

print ("Estimated Makespan:", fitness_fun(random_Allocation))
map_cluster_Size(random_Allocation)

''' To write the estimated Makespan in file'''
with open('rand_Allocation_est_makespan.txt', 'a') as f:
    f.write("%s\n" % str(fitness_fun(random_Allocation)))
