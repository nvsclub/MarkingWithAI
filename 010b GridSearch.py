import lib.marking_evaluation as meval

from tqdm import tqdm
import pandas as pd
from time import time, asctime

# Define adversary
adversary = meval.default_adversary_1

# Create register to save information about the run
register = {'proposal': [], 'fitness': [], 'cycle_time': []}

# Define number of runs
all_solutions = [[[i, j] for i in range(100) for _ in range(10)] for j in range(100)]

# Perform n_runs
for proposed_team in tqdm(range(all_solutions)):
    # Starts timer
    start_time = time()

    # Evaluating randomly created team
    fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

    # Save results for post-hoc analysis
    register['proposal'].append(proposed_team)
    register['fitness'].append(fitness)
    register['cycle_time'].append(time() - start_time)
    
# Export registers to CSV
export_time = asctime().replace(':','').replace(' ','')
pd.DataFrame(register).to_csv(f'results/gridsearch_da{export_time}.csv', index=False)
