import lib.marking_evaluation as meval
from random import randint

from tqdm import tqdm
import pandas as pd
from time import time, asctime

# Define adversary
adversary = meval.default_adversary_1

# Create register to save information about the run
register = {'proposal': [], 'fitness': [], 'cycle_time': []}

# Define number of runs
n_runs = 25000
all_solutions = [[[i, j] for i in range(100) for _ in range(10)] for j in range(100)]

# Perform n_runs
for _ in tqdm(range(n_runs)):
    # Starts timer
    start_time = time()

    # Generating a random team
    proposed_team = randint(0, len(all_solutions) - 1)

    # Evaluating randomly created team
    fitness = adversary.calculate_heuristic(meval.create_team(all_solutions[proposed_team]))

    # Save results for post-hoc analysis
    register['proposal'].append(proposed_team)
    register['fitness'].append(fitness)
    register['cycle_time'].append(time() - start_time)

    # Remove team from possible solutions to evaluate
    del all_solutions[proposed_team]
    
# Export registers to CSV
export_time = asctime().replace(':','').replace(' ','')
pd.DataFrame(register).to_csv(f'results/random_search_da{export_time}.csv', index=False)
