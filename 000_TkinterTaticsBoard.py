# Imports
import tkinter
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import lib.draw as draw

import numpy as np
from random import random

# Initializing Tkinter
root = tkinter.Tk()
root.geometry('1512x980')
root.configure(background='white')
root.wm_title("Embedding in Tk")

global team1_selected
team1_selected = True
global team_1_players
global team_2_players
team_1_players = []
team_2_players = []

def update_screen():
    global team_1_players
    global team_2_players

    plt.figure()

    draw.pitch()

    for coordinates in team_1_players:
        plt.scatter(coordinates[0], coordinates[1], s=100, color='navy', edgecolors='navy', alpha=0.7, linewidths=1,  zorder=100)

    for coordinates in team_2_players:
        plt.scatter(coordinates[0], coordinates[1], s=100, color='firebrick', edgecolors='firebrick', alpha=0.7, linewidth=1, zorder=100)

    plt.savefig('_tmp/b.png')
    plt.clf()

    img = ImageTk.PhotoImage(Image.open('_tmp/b.png'))
    panel.configure(image=img)
    panel.image = img

# Retrieve mouse information and display it on screen
def get_clicks(event):
    global team_1_players
    global team_2_players

    # Standardize coordinates
    xt = (event.x - 241) / 10.70
    yt = ((event.y) * -1 + 841) / 6.90 # Inverting coordinates in the end

    if team1_selected:
        team_1_players.append((xt, yt))
    else:
        team_2_players.append((xt, yt))

    update_screen()

def select_team1(event):
    global team1_selected
    
    team1_selected = True

def select_team2(event):
    global team1_selected
    
    team1_selected = False

def print_positions(event):
    global team_1_players
    global team_2_players
    
    print(team_1_players, team_2_players)

def reset(event):
    global team_1_players
    global team_2_players

    team_1_players = []
    team_2_players = []

    update_screen()


def _quit():
    root.quit()
    root.destroy()

draw.pitch()
plt.savefig('_tmp/aaa.png')

root.bind('1', select_team1)
root.bind('2', select_team2)
root.bind('r', reset)
root.bind('p', print_positions)

img = ImageTk.PhotoImage(Image.open('_tmp/aaa.png'))
panel = tkinter.Label(root, image=img)
panel.bind('<Button-1>', get_clicks)
panel.place(anchor='nw', x=0, y=0)

root.protocol("WM_DELETE_WINDOW", _quit)

# Enter mainloop
tkinter.mainloop()