import lib.marking_evaluation as meval

from tqdm import tqdm
import pandas as pd
from time import time, asctime

# Define adversary
adversary = meval.default_adversary_1

# Create register to save information about the run
register = {'proposal': [], 'fitness': [], 'cycle_time': []}

# Define number of runs
n_runs = 25000

# Perform n_runs
for _ in tqdm(range(n_runs)):
    # Starts timer
    start_time = time()

    # Generating a random team
    proposed_team = meval.generate_random_start()

    # Evaluating randomly created team
    fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

    # Save results for post-hoc analysis
    register['proposal'].append(proposed_team)
    register['fitness'].append(fitness)
    register['cycle_time'].append(time() - start_time)
    
# Export registers to CSV
pd.DataFrame(register).to_csv(f'results/random_search_da{asctime()}.csv', index=False)
