''' Google Or-Tool Implementation.'''
import collections
from numpy.random import normal
from numpy import rint
import random
import time
from ortools.linear_solver import pywraplp

''' Systems and task allocation setup '''
NUM_TASKS = 45    # n tasks to be scheduled
#task_type = [0] * NUM_TASKS  # All tasks have the same type
task_type = [0,1,2] * NUM_TASKS # Different tasks
print ("Tasks Types=", task_type)
max_task_id=max(task_type)
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
''' define individual node capacities '''
n = NUM_TASKS
num_workers = len(new_costs)
node_cap = [n*0.2]*3 + [n*0.2]*4 + [n*0.7]*1;
node_cap = [round(i) for i in node_cap]
print ("Node_cap=", node_cap)
############ end cluster and task setup

''' solver setup '''
start = time.time()
solver = pywraplp.Solver('SolveAssignmentProblem',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# x vairable. if task is allocated to a node. x=1. otherwise, x=0
x = {}
for i in range(num_workers):
    for j in range(NUM_TASKS):
        x[i,j] = solver.IntVar(0,1,'')

## one entry per node if there is any task allocated to this node.
tasks_on_node = []
for taskId in range (0,max_task_id+1):
    tasks_on_node.append([])
    for i in range (num_workers):
        tasks_on_node[taskId].append(solver.IntVar(0,1,''))

# constraint 1: number of tasks assigned to each node
#               must be less than node capacity
for i in range(num_workers):
    solver.Add(solver.Sum([x[i,j] for j in range(NUM_TASKS)]) <= node_cap[i])

# constraint 2: each task is assigned to exactly one node
for j in range(NUM_TASKS):
    solver.Add(solver.Sum([x[i,j] for i in range(num_workers)]) == 1)

# constraint 3:
for taskId in range(0,max_task_id+1):
    mask = [(1 if (task_type[j] == taskId) else 0) for j in range(NUM_TASKS)]
    for i in range(num_workers):
        for j in range (NUM_TASKS):
            solver.Add(tasks_on_node[taskId][i] >= x[i,j] * mask[j])

times = []  #index on each worker node
for taskId in range(0,max_task_id+1):
    times.append([])
    mask = [(1 if (task_type[j] == taskId) else 0) for j in range(NUM_TASKS)]
    for node in range(num_workers):
        m = new_costs[node][taskId]['m']
        c = new_costs[node][taskId]['c']
        times[taskId].append(m * solver.Sum([x[node,j] * mask[j] for j in range(NUM_TASKS)]) +
                        (tasks_on_node[taskId][node] * c))

## Minimize ( max(x,y) ) becomes z > x, z>y -> 2 new constraints.
## Reformulting the max in linear model.
z = solver.NumVar(0,solver.infinity(), 'z')
for taskId in range(0,max_task_id+1):
    for i in range(num_workers):
        #solver.Add(z >= times[taskId][i])
        solver.Add(z >= solver.Sum([times[taskId][i] for taskId in range(0,max_task_id+1)]))
solver.Minimize(z)
status = solver.Solve()
print('Estimated makespan= ', solver.Objective().Value())
print('Number of constraints =', solver.NumConstraints())

''' To write the estimated Makespan in file'''
with open('mip_est_makespan.txt', 'a') as f:
    f.write("%s\n" % str(solver.Objective().Value()))

################################################################################
'''To map ip address of nodes'''
edge_devices = ['192.168.1.229', '192.168.1.222', '192.168.1.224',
                '192.168.1.148', '192.168.1.230', '192.168.1.168', '192.168.1.182',
                '192.168.1.131']

new_edge_devices = []
final_Workers_IP=[0]*NUM_TASKS
e_devices='('

''' Printing and mapping ip addresses of nodes '''
if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    print ("--- mapping nodes to tasks---");
    for i in range(num_workers):
        for j in range(NUM_TASKS):
            if x[i,j].solution_value() > 0.5:
                final_Workers_IP[j]='\''+edge_devices [i]+'\' '
                e_devices+='\''+edge_devices [i]+'\' '
                print('edge node %d assigned to task %d' % (i, j))
    e_devices = e_devices[:-1]
    e_devices+=')'
    finalIPsBashFormat='('
    for i in range(NUM_TASKS):
        finalIPsBashFormat+=final_Workers_IP[i]
    finalIPsBashFormat= finalIPsBashFormat[:-1]
    finalIPsBashFormat+=')'
    print ()
    print("Nodes Order:", finalIPsBashFormat)
    ''' To wirte results and map ip address to nodes'''
    with open('mip_allocation.txt', 'w') as f:
        f.write(str(finalIPsBashFormat))
    ''' To write the estimated Makespan in file'''
else:
    print('No solution found.')
end = time.time()
allocation_Time=((end - start))
print("AllocationTime = ", allocation_Time, "seconds")
