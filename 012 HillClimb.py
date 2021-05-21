import lib.marking_evaluation as meval
import copy

from time import time
import pandas as pd

# Lists all the sucessors of a team proposal
def list_successors(current_proposal):
    successors = []

    # This chained fors generate every possible movement for all the players
    # according to the step globally defined
    for i in range(10):
        for j in range(2):
            for k in [-step, step]:
                changed_proposal = copy.deepcopy(current_proposal)
                changed_proposal[i][j] += k
                successors.append(changed_proposal)
    
    # Returns a list with all the successors of the team proposal
    return successors

# Finds the best proposal from a list
def find_best(adversary, successors):
    # Assuming the first "best" is the first element of the list
    best = successors[0]
    best_heuristic = adversary.calculate_heuristic(meval.create_team(best))

    # Calculating the fitness for all successors
    for successor in successors[1:]:
        successor_heuristic = adversary.calculate_heuristic(meval.create_team(successor))
        
        # If sucessor is the best, save the result 
        if successor_heuristic > best_heuristic:
            best = successor
            best_heuristic = successor_heuristic
    
    # Returns the best proposal and the heuristic
    return best, best_heuristic

# Define adversary
adversary = meval.default_adversary_1

# Create register to save information about the run
register = {'proposal': [], 'fitness': [], 'cycle_time': []}

# Starting from a random position
proposed_team = meval.generate_random_start()
best_fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

# Defining step related variables
step = 10
step_decrease_cycle = 125
counter = 0

# Search for a better solution until:
# No better solution can be found and step cannot be lowered
while 1:
    start_time = time()
    
    # Halve step every *step_decrease_cycle* iterations
    counter += 1
    if counter % step_decrease_cycle == 0:
        step = int(step/2)

    # Find the best sucessor
    successors = list_successors(proposed_team)
    proposed_team, fitness = find_best(adversary, successors)

    # Save results for post-hoc analysis
    register['proposal'].append(proposed_team)
    register['fitness'].append(fitness)
    register['cycle_time'].append(time() - start_time)

    # If fitness did not improve
    if fitness < best_fitness:
        # If step is already one, end optimization
        if step == 1:
            break
        
        # If we can decrease the step, do it before ending
        else:
            counter = 0
            step = int(step/2)

        # Export registers to CSV
        pd.DataFrame(register).to_csv('results/hillclimb_da2.csv', index=False)
    
    # If fitness improved, continue searching
    else:        
        best_fitness = fitness

    print(counter, end='\r')

# Export registers to CSV
pd.DataFrame(register).to_csv('results/hillclimb_da2.csv', index=False)
