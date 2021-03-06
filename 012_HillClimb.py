import lib.marking_evaluation as meval
import copy

from time import time, asctime
from tqdm import tqdm
import pandas as pd

# Lists all the sucessors of a team proposal
def list_successors(current_proposal, step):
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

# Calculate sucessor fitness
def calculate_sucessor_fitnesses(adversary, successors):
    # Calculating the fitness for all successors
    successor_fitnesses = []
    for successor in successors:
        successor_fitnesses.append(adversary.calculate_heuristic(meval.create_team(successor)))

    # Returns list with the fitness of all successors
    return successor_fitnesses

# Finds the best proposal from a list
def find_best(successors, successor_fitnesses):
    # Assuming the first "best" is the first element of the list
    best = successors[0]
    best_fitness = successor_fitnesses[0]

    # Calculating the fitness for all successors
    for successor, fitness in zip(successors[1:], successor_fitnesses[1:]):
        # If sucessor is the best, save the result 
        if fitness > best_fitness:
            best = successor
            best_fitness = fitness
    
    # Returns the best proposal and the heuristic
    return best, best_fitness

# Hill climb algorithm
def hillclimb():
    # Create register to save information about the run
    register = {'iteration': [], 'proposal': [], 'fitness': [], 'cycle_time': []}

    # Starting from a random position
    proposed_team = meval.generate_random_start(adversary.x_min, adversary.x_max)
    best_fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

    # Defining step related variables
    step = 10
    step_decrease_cycle = 125
    max_steps_at_current_step = 250
    counter = 0
    internal_counter = 0

    # Search for a better solution until:
    # No better solution can be found and step cannot be lowered
    while 1:
        counter += 1
        
        # Reset timer
        start_time = time()
        
        # Halve step every *step_decrease_cycle* iterations
        internal_counter += 1
        if internal_counter % step_decrease_cycle == 0:
            internal_counter = 0
            step = int(step/2)

        # Find the best sucessor
        successors = list_successors(proposed_team, step)
        successor_fitnesses = calculate_sucessor_fitnesses(adversary, successors)
        proposed_team, fitness = find_best(successors, successor_fitnesses)

        # If fitness did not improve
        if fitness <= best_fitness:
            # If step is already one, end optimization
            if step == 1:
                break
            
            # If we can decrease the step, do it before ending
            else:
                internal_counter = 0
                step = int(step/2)
        
        # If fitness improved
        else:
            # If the hill climb is stuck at the minimum temperature, break
            if internal_counter > max_steps_at_current_step:
                break

            # Else, save the result and continue
            best_fitness = fitness

        # Save results for post-hoc analysis
        for successor, successor_fitness in zip(successors, successor_fitnesses):
            register['iteration'].append(counter)
            register['proposal'].append(successor)
            register['fitness'].append(successor_fitness)
            register['cycle_time'].append(time() - start_time)

    # Export registers to CSV
    current_time = asctime().replace(':','_').split(' ')
    export_time = f'{current_time[1]}_{current_time[2]}_{current_time[3]}'
    pd.DataFrame(register).to_csv(f'results/012_hillclimb_{export_time}.csv', index=False)

# Define adversary
adversary = meval.default_adversary_1

# Run algorithm multiple times
for _ in tqdm(range(20)):
    hillclimb()