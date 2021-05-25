import lib.marking_evaluation as meval
import copy
from random import random

import pandas as pd
import numpy as np
from time import time, asctime

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

# Simulated annealing algorithm
def simulatedannealing():
    # Create register to save information about the run
    register = {'proposal': [], 'fitness': [], 'cycle_time': [], 'temperature': []}

    # Starting from a random position
    proposed_team = meval.generate_random_start()
    fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

    # Defining step related variables
    step = 10
    step_decrease_cycle = 125
    max_steps_at_min_temperature = 250
    counter = 0

    # Initializing variables
    temperature = 1
    best_fitness = fitness
    probability_of_acceptance = 0
    counter = 0
    counter_min_temp = 0
    start_time = time()

    # Search for a better solution until:
    # Temperature < minimum_temperature and no better solution can be found
    while 1:
        counter += 1

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
        register['temperature'].append(temperature)

        # Check for end clauses
        if temperature < minimum_temperature:
            # Break if no improvement is possible
            if best_fitness > fitness:
                break

            # Break if there are too many steps at minimum temperature
            counter_min_temp += 1
            if counter_min_temp > max_steps_at_min_temperature:
                break
        
        # Check if temperature enables randomization
        elif temperature > random():
            # Generate and evaluate random proposal
            random_proposal = meval.generate_random_start()
            random_proposal_heuristic = adversary.calculate_heuristic(meval.create_team(random_proposal))
            
            # Calculate probability of acceptance (note: sigmoid not good)
            probability_of_acceptance = ((np.arctan((random_proposal_heuristic - fitness)*100) + np.pi/2) / np.pi) * temperature
            
            # If proposal is accepted, update new proposal and reset step
            if probability_of_acceptance > random():
                proposed_team = random_proposal
                fitness = random_proposal_heuristic
                step = 10

        # If solution cannot improved, decrease step, else generate a new random point
        elif best_fitness >= fitness:
            # If step is already one, end optimization
            if step == 1:
                proposed_team = meval.generate_random_start()
            
            # If we can decrease the step, do it before ending
            else:
                counter = 0
                step = int(step/2)

        # If solution improved, register new best fitness
        elif fitness > best_fitness:
            best_fitness = fitness

        # Update temperatire
        temperature *= temperature_multiplier

        print(counter, end='\r')
        
    # Export registers to CSV
    export_time = asctime().replace(':','').replace(' ','')
    pd.DataFrame(register).to_csv(f'results/simulatedannealing_da{export_time}.csv', index=False)

# Define adversary
adversary = meval.default_adversary_1

# Defining parameters for the algorithm
temperature_multiplier = 0.96
minimum_temperature = 0.01

for _ in range(4):
    simulatedannealing()