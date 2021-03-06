import lib.marking_evaluation as meval

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from time import time, asctime
from tqdm import tqdm
import pandas as pd

# Performs a study using the optuna library
def do_study(adversary, sampler, n_runs=1000):

    # Defining objective function
    def objective(trial):
        # Start time needs to be global due to hidden optimization costs
        global start_time

        # Organize all optuna suggestions
        proposed_team = []
        for i in range(10):
            proposed_team.append([trial.suggest_int('x'+str(i), adversary.x_min, adversary.x_max), trial.suggest_uniform('y'+str(i), 0, 100)])

        # Calculating fitness        
        fitness = adversary.calculate_heuristic(meval.create_team(proposed_team))
    
        # Save results for post-hoc analysis
        register['proposal'].append(proposed_team)
        register['fitness'].append(fitness)
        register['cycle_time'].append(time() - start_time)

        # Starting clock after registering to count for hidden optimization costs
        start_time = time()

        # Return fitness for optimization
        return fitness

    # Define study and perform n_runs
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_runs)

# Define adversary
adversary = meval.default_adversary_1

# Initialize global timer
global start_time
start_time = time()

# Define number of runs
n_runs = 5000

# Run algorithm multiple times
for _ in tqdm(range(20)):
    # Create register to save information about the run
    register = {'proposal': [], 'fitness': [], 'cycle_time': []}
    start_time = time()

    # Perform study (CMA-ES)
    do_study(adversary, optuna.samplers.CmaEsSampler(), n_runs=n_runs)

    # Export registers to CSV
    current_time = asctime().replace(':','_').split(' ')
    export_time = f'{current_time[1]}_{current_time[2]}_{current_time[3]}'
    pd.DataFrame(register).to_csv(f'results/011a_cmaes_{export_time}.csv', index=False)

    # Reset register
    register = {'proposal': [], 'fitness': [], 'cycle_time': []}
    start_time = time()

    # Perform study (TPE)
    do_study(adversary, optuna.samplers.TPESampler(), n_runs=n_runs)

    # Export registers to CSV
    current_time = asctime().replace(':','_').split(' ')
    export_time = f'{current_time[1]}_{current_time[2]}_{current_time[3]}'
    pd.DataFrame(register).to_csv(f'results/011b_tpe_{export_time}.csv', index=False)