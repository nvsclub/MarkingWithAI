import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2

import lib.marking_evaluation as meval

def plot_team(team, save_fig_dir='', dpi=120):
    adversary = meval.default_adversary_1
    team = [float(i) for i in team.replace('[','').replace(']','').replace(' ','').split(',')]
    aux = []
    for i in range(10):
        aux.append([team[2*i], team[2*i+1]])
    team = meval.create_team(aux)

    adversary.plot_result(team, save_fig_dir=save_fig_dir, dpi=dpi)

for file_name in tqdm(glob('results/*.csv')):
    file_name = file_name.replace('results\\','').replace('.csv','')
    res_criteria_1 = pd.read_csv(f'results/{file_name}.csv')
    video = cv2.VideoWriter(f'img/animated/{file_name}.mp4',cv2.VideoWriter_fourcc(*'MPEG'),1,(1247,803))

    running_fitness = 0
    for i, row in res_criteria_1.iterrows():
        if (row.fitness > running_fitness * 1.01) or (i == len(res_criteria_1) - 1):
            plot_team(row.proposal, save_fig_dir=f'_tmp/giftmp.png')
            video.write(cv2.imread(f'_tmp/giftmp.png'))
            running_fitness = row.fitness

video.release()