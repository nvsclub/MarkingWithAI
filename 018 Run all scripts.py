import os

def run_script(file):
    print(f'Running {file}')
    try: os.system(f'python \"{file}\"')
    except: print(f'Error occured in {file}')

run_script('010 RandomSearch.py')
run_script('011 Optuna.py')
run_script('012 HillClimb.py')
run_script('013 SimulatedAnnealing.py')
run_script('014 GeneticAlgorithms.py')