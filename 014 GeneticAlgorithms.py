import lib.marking_evaluation as meval
import numpy as np
import copy
from random import randint, random, choices

from time import time, asctime
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
def selection(population, population_fitnesses):
    # Calculate which members survive automatically
    survival_guaranteed_above = np.quantile(population_fitnesses, 1 - elitist_rate)

    # Find which individuals are unfit
    unfit_individuals = []
    i = 0
    for fitness in population_fitnesses:
        if fitness < survival_guaranteed_above:
            if survival_rate < random():
                unfit_individuals.append(i)
        i+=1

    # Remove unfit individuals
    for individual_id in reversed(unfit_individuals):
        del population[individual_id]
        del population_fitnesses[individual_id]
    
    # Returns selected population
    return population, population_fitnesses

# Crossover the population to generate new individuals
def crossover(population, population_fitnesses):
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
        individual = mutation(individual)

        # Add individual to the population
        population.append(individual)
        population_fitnesses.append(0) # Keep size consistent

    # Returns filled population
    return population

# Mutate members of the population
def mutation(individual):
    # Randomly generate all players to be mutated
    mutated_players = [randint(0, 9) for i in range(randint(0, 9))]

    # Actually mutate the players by adding a random player to a position
    for player in mutated_players:
        if random() < mutation_rate:
            individual[player] = [randint(0, 100), randint(0, 100)]

    # Return mutated population
    return individual

# Genetic algorithm
def geneticalgorithm():
    # Create register to save information about the run
    register = {'proposal': [], 'fitness': [], 'cycle_time': []}

    # Initializing population
    population = [meval.generate_random_start() for _ in range(population_size)]

    # Initializing variables
    best_heuristic = -100
    iterations_without_improving = 0
    start_time = time()

    # Run the generations
    for _ in range(limit_of_generations):
        # Evaluate population
        population_fitnesses = evaluate(population)

        # Verify if any member of the population improved the max fitness 
        fitness = max(population_fitnesses)
        if fitness > best_heuristic:
            best_heuristic = fitness
            iterations_without_improving = 0

        # Save results for post-hoc analysis
        register['proposal'].append(population[population_fitnesses.index(fitness)])
        register['fitness'].append(fitness)
        register['cycle_time'].append(time() - start_time)
        start_time = time()
        
        # Select members that survive
        population, population_fitnesses = selection(population, population_fitnesses)
        # Generate new members of the population and mutate
        population = crossover(population, population_fitnesses)

        # Increase counter
        iterations_without_improving += 1
        if iterations_without_improving > no_stable_generations or fitness > fitness_limit:
            break

    # Export registers to CSV
    export_time = asctime().replace(':','').replace(' ','')
    pd.DataFrame(register).to_csv(f'results/genetic_da{export_time}.csv', index=False)

# Define adversary
adversary = meval.default_adversary_1

# Defining GA parameters
population_size = 100
fitness_limit = 10
limit_of_generations = 300
no_stable_generations = 10
elitist_rate = 0.05
survival_rate = 0.7
mutation_rate = 0.2

for _ in range(5):
    geneticalgorithm()