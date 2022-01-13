
from numpy.random import normal
from numpy import rint
import random
import time
from ortools.linear_solver import pywraplp

def main():
    #-------------------------------------------
  #   #randomize code created by Jeremy;
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

    ##########

    print('final x:')
    print('----------')
    prMatrix(x)

  #-------------------------------------------
    #begin Google-or Tool;
    # Data
    costs = x

    print (costs)
    print ("\n")

    num_workers = len(costs)
    num_tasks = len(costs[0])

    #node_cap = [(n),(n),(n),(n), (n),(n),(n),(n), (n),(n)] #equal nodes
    node_cap = [
                (n*0.05), (n*0.05), (n*0.05), (n*0.05), (n*0.05), (n*0.05), (n*0.05), (n*0.05), (n*0.05),
                (n*0.1),  (n*0.1), (n*0.1),  (n*0.1), (n*0.1),  (n*0.1), (n*0.1),  (n*0.1), (n*0.1),
                (n*0.6), (n*0.6),
                ]

    print (node_cap)

    # Solver
    # Create the mip solver with the SCIP backend.
    #solver = pywraplp.Solver.CreateSolver('MIP')

    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    start = time.time()
    edge_devices = ['pi2', 'pi2', 'pi2', 'pi2', 'pi2', 'pi2', 'pi2', 'pi2', 'pi2',
                    'pi3', 'pi3', 'pi3', 'pi3', 'pi3', 'pi3', 'pi3', 'pi3', 'pi3',
                    'pi4', 'pi4']

    new_edge_devices = []
    e_devices='('

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, '')

    # Constraints
    # Number of tasks assinged to each node less than the node capacitiy!
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= node_cap[i])

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])

    solver.Minimize(solver.Sum(objective_terms))
    #solver.Maximize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()
    print('Minimum cost = ', solver.Objective().Value())

    #print()
    final_Workers_IP=[0]*len(costs[1])
    #print()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        #print('Total cost = ', solver.Objective().Value(), '\n')
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
        #print (new_edge_devices)
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
