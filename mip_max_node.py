''' Google Or-Tool Implementation.
    This new version is to Optimise Task Allocation for Edge Micro-clusters.'''
from numpy.random import normal
from numpy import rint
import random
import time
from ortools.linear_solver import pywraplp
def main():
    def prMatrix(x):
        for row in x:
            for val in row:
                print(val,end=',')
            print()
        print()
    n= 30 #number of tasks to be allocated
    ''' Estimated extecution time tasks take on nodes.
        50 means Pi2B, 40 means Pi3B, 10 means Pi4B '''
    x = [
        [50]*n, [50]*n, [50]*n,                 ## Pi2B
        [40]*n, [40]*n, [40]*n, [40]*n,         ## Pi3B
        [10]*n                                  ## Pi4B
        ]
    print('Initial tasks execution times:')
    prMatrix(x)
    ''' Code to randomize data created by Jeremy '''
    ####### @jsinger new perturbation code for cost matrix
    ####### thresholds for changing costs
    COEFFICIENT_OF_VARIATION=0.25  # c.o.v. = stdev / mean = sigma/mu
    for i in range(len(x)):
        for j in range(len(x[i])):
            mu = x[i][j]
            sigma = COEFFICIENT_OF_VARIATION * mu
            updated_value = int(rint(normal(mu, sigma)))
            x[i][j] = max(0, updated_value)  # no negative costs!
    print('Estimated tasks execution time:')
    prMatrix(x)
    #--------------#begin Google-or Tool;---------------------------------------
    ''' Google Or-tool sovler begin '''
    start = time.time()
    costs = x
    num_workers = len(costs)
    num_tasks = len(costs[0])

    ''' define node capabilites '''
    node_cap = [n*0.05]*3 + [n*0.1]*4 + [n*0.6]*1

    '''Solver
       Create the mip solver with the SCIP backend.
       solver = pywraplp.Solver.CreateSolver('MIP')'''

    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    ''' We could inseart nodes IP address here inseated of nodes names'''
    edge_devices = ['pi2', 'pi2', 'pi2',
                    'pi3', 'pi3', 'pi3', 'pi3+',
                    'pi4']

    new_edge_devices = []
    e_devices='('

    '''Variables
       x[i, j] is an array of 0-1 variables, which will be 1
       if worker i is assigned to task j. '''
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, '')

    ''' Constraints 1
        Number of tasks assinged to each node less than the node capacitiy!'''
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= node_cap[i])

    ''' Constraints 2
        Each task is assigned to exactly one worker.'''
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    ''' Objective '''
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))
    status = solver.Solve()
    print('Minimum cost = ', solver.Objective().Value())

    ''' To Return node with maximum execution time (new makespan) '''
    def findmax():
        excTime = []
        bestSolution = []
        tasks = []
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(num_workers):
                for j in range(num_tasks):
                    if x[i, j].solution_value() > 0.5:
                        bestSolution.append(i)
                        tasks.append(j)
                        excTime.append(costs [i][j])
        sums = {}
        for key, value in zip(bestSolution,excTime):
            try:
                sums[key] += value
            except KeyError:
                sums[key] = value
        maximum = max(sums, key=sums.get)
        print ("Best Nodes Allocation: ", bestSolution)
        print ("Tasks ID: ", tasks)
        print ("Tasks excTimes: ", excTime)
        print("Nodes execution Time: ", sums)
        print("Node ID:", maximum, "has MaxExecutionTime:", sums[maximum])
        print ("")
    findmax()
    ##----------End-------------------------------------------------------------
    ''' Printing solution '''
    final_Workers_IP=[0]*len(costs[1])
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        #print('Total cost = ', solver.Objective().Value(), '\n')
        print ("Mapping Nodes to Tasks:")
        print ("-----------------------")
        for i in range(num_workers):
            for j in range(num_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    final_Workers_IP[j]='\''+edge_devices [i]+'\' '
                    e_devices+='\''+edge_devices [i]+'\' '
                    print('Edge node %d assigned to task %d.  Cost = %d' % (i, j, costs[i][j]))
        print()
        end = time.time()
        print("Time = ", round(end - start, 4), "seconds")
        e_devices = e_devices[:-1]
        e_devices+=')'
        finalIPsBashFormat='('
        for i in range(num_tasks):
            finalIPsBashFormat+=final_Workers_IP[i]
        finalIPsBashFormat= finalIPsBashFormat[:-1]
        finalIPsBashFormat+=')'
        print ()
        print(finalIPsBashFormat)
if __name__ == '__main__':
    main()
