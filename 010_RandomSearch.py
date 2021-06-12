import lib.marking_evaluation as meval
from random import randint

from tqdm import tqdm
import pandas as pd
from time import time, asctime

# Define adversary
adversary = meval.default_adversary_1

# Define number of runs
n_runs = 20000

# Run algorithm multiple times
for _ in tqdm(range(20)):
    # Create register to save information about the run
    register = {'proposal': [], 'fitness': [], 'cycle_time': []}

    # Perform n_runs
    for _ in range(n_runs):
        # Starts timer
        start_time = time()

        # Generating a random team
        proposed_team = meval.generate_random_start(adversary.x_min, adversary.x_max)

        # Evaluating randomly created team
        fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))

        # Save results for post-hoc analysis
        register['proposal'].append(proposed_team)
        register['fitness'].append(fitness)
        register['cycle_time'].append(time() - start_time)

    # Export registers to CSV
    current_time = asctime().replace(':','_').split(' ')
    export_time = f'{current_time[1]}_{current_time[2]}_{current_time[3]}'
    pd.DataFrame(register).to_csv(f'results/010_randsearch_{export_time}.csv', index=False)
