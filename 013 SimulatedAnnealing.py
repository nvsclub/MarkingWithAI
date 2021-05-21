import lib.marking_evaluation as meval
import copy
from random import random

import pandas as pd
from time import time

# Lists all the sucessors of a team proposal
def list_successors(current_proposal):
    successors = []

    # This chained fors generate every possible movement for all the players
    # according to the step globally defined
    for i in range(10):
        for j in range(2):
            for k in [-1, 1]:
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
register = {'proposal': [], 'fitness': [], 'cycle_time': [], 'temperature': []}

# Starting from a random position
proposed_team = meval.generate_random_start()
fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

# Defining parameters for the algorithm
temperature = 1
temperature_multiplier = 0.98
minimum_temperature = 0.001

# Initializing variables
best_fitness = fitness
probability_of_acceptance = 0
counter = 0
start_time = time()

# Search for a better solution until:
# Temperature < minimum_temperature and no better solution can be found
while 1:
    counter += 1

    # Find the best sucessor
    successors = list_successors(proposed_team)
    proposed_team, fitness = find_best(adversary, successors)

    # Save results for post-hoc analysis
    register['proposal'].append(proposed_team)
    register['fitness'].append(fitness)
    register['cycle_time'].append(time() - start_time)
    register['temperature'].append(temperature)

    # Check for end clauses
    if temperature < minimum_temperature:
        # Break if no improvement is possible
        if best_fitness > fitness:
            break

        # Export registers to CSV
        if counter % 100:
            pd.DataFrame(register).to_csv('results/hillclimb_da1.csv', index=False)
    
    # Check if temperature enables randomization
    elif temperature > random():
        # Generate and evaluate random proposal
        random_proposal = meval.generate_random_start()
        random_proposal_heuristic = adversary.calculate_heuristic(meval.create_team(random_proposal))
        
        # Calculate probability of acceptance (note: sigmoid not good)
        probability_of_acceptance = 10 ** ((random_proposal_heuristic - fitness)) * temperature
        
        # If proposal is accepted, update new proposal
        if probability_of_acceptance > random():
            proposed_team = random_proposal
            fitness = random_proposal_heuristic

    # If solution cannot improved, generate a new random point
    elif best_fitness > fitness:
        proposed_team = meval.generate_random_start()

    # If solution improved, register new best fitness
    elif fitness > best_fitness:
        best_fitness = fitness

    # Update temperatire
    temperature *= temperature_multiplier

    print(counter, end='\r')
    
# Export registers to CSV
pd.DataFrame(register).to_csv('results/simulatedannealing_da1.csv', index=False)