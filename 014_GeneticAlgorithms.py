import lib.marking_evaluation as meval
import numpy as np
import copy
from random import randint, random, choices

from time import time, asctime
from tqdm import tqdm
import pandas as pd

# Evaluate all population individuals
def evaluate(population):
    # Calculate fitness for all individuals
    population_fitnesses = []
    for individual in population:
        population_fitnesses.append(adversary.calculate_heuristic(meval.create_team(individual)))
    
    # Returns list with all population fitnesses
    return population_fitnesses

# Select which members go through to the next round
def selection(population, population_fitnesses, current_survival_rate=0.7):
    # Calculate which members survive automatically
    survival_guaranteed_above = np.quantile(population_fitnesses, 1 - elitist_rate)

    # Find which individuals are unfit
    unfit_individuals = []
    i = 0
    for fitness in population_fitnesses:
        if fitness < survival_guaranteed_above:
            if current_survival_rate < random():
                unfit_individuals.append(i)
        i+=1

    # Remove unfit individuals
    for individual_id in reversed(unfit_individuals):
        del population[individual_id]
        del population_fitnesses[individual_id]
    
    # Returns selected population
    return population, population_fitnesses

# Crossover the population to generate new individuals
def crossover(population, population_fitnesses, current_mutation_rate=0.2):
    # Fill the population to its size
    while len(population) < population_size:
        # Randomly select 2 parents, weights are fitnesses
        parents = choices(population=population, weights=[np.exp(fitness) for fitness in population_fitnesses], k=2)

        # Create the new individual as a copy of the first parent
        individual = copy.deepcopy(parents[0])
        
        # Randomly choose which players from the second parent will be used
        crossover_players = [randint(0, 9) for _ in range(randint(1, 8))]

        # Add players from the second parent
        for player in crossover_players:
            individual[player] = parents[1][player]

        # Add mutations to the individual
        individual = mutation(individual, current_mutation_rate)

        # Add individual to the population
        population.append(individual)
        population_fitnesses.append(0) # Keep size consistent

    # Returns filled population
    return population

# Mutate members of the population
def mutation(individual, current_mutation_rate):
    # Randomly generate all players to be mutated
    mutated_players = [randint(0, 9) for _ in range(randint(0, 10))]

    # Actually mutate the players by adding a random player to a position
    for player in mutated_players:
        if random() < current_mutation_rate:
            individual[player] = [randint(adversary.x_min, adversary.x_max), randint(0, 100)]

    # Return mutated population
    return individual

# Genetic algorithm
def geneticalgorithm():
    # Create register to save information about the run
    register = {'iteration': [], 'proposal': [], 'fitness': [], 'cycle_time': []}

    # Initializing population
    population = [meval.generate_random_start(adversary.x_min, adversary.x_max) for _ in range(population_size)]

    # Initializing variables
    best_heuristic = -100
    iterations_without_improving = 0
    start_time = time()

    # Run the generations
    for i in range(limit_of_generations):
        # Evaluate population
        population_fitnesses = evaluate(population)

        # Verify if any member of the population improved the max fitness 
        fitness = max(population_fitnesses)
        if fitness > best_heuristic:
            best_heuristic = fitness
            iterations_without_improving = 0

        # Save results for post-hoc analysis
        for individual, individual_fitness in zip(population, population_fitnesses):
            register['iteration'].append(i)
            register['proposal'].append(individual)
            register['fitness'].append(individual_fitness)
            register['cycle_time'].append(time() - start_time)
        start_time = time()
        
        # Select members that survive
        population, population_fitnesses = selection(population, population_fitnesses, 
            current_survival_rate=default_survival_rate / np.exp(iterations_without_improving/no_stable_generations))
        # Generate new members of the population and mutate
        population = crossover(population, population_fitnesses,
            current_mutation_rate=default_mutation_rate * np.exp(iterations_without_improving/no_stable_generations))

        # Increase counter
        iterations_without_improving += 1
        if iterations_without_improving > no_stable_generations or fitness > fitness_limit:
            break

    # Export registers to CSV
    current_time = asctime().replace(':','_').split(' ')
    export_time = f'{current_time[1]}_{current_time[2]}_{current_time[3]}'
    pd.DataFrame(register).to_csv(f'results/014_genetic_{export_time}.csv', index=False)

# Define adversary
adversary = meval.default_adversary_1

# Defining GA parameters
population_size = 100
fitness_limit = 10
limit_of_generations = 300
no_stable_generations = 20
elitist_rate = 0.05
default_survival_rate = 0.7
default_mutation_rate = 0.2

for _ in tqdm(range(20)):
    geneticalgorithm()