#TODO Known Bug: current framework cannot calculate if vertical passline is intersected
## because of division by 0 in linear_equation_from_2_points

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lib.draw as draw
from sklearn.neighbors import KNeighborsClassifier
from random import random, randint

# Distance
## Requires two points (a and b).
## Returns distance between the two points.
def distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# Distance to line
## Requires three points (p0, p1, p2)
## Returns the distance from player p0 to the line defined by players (p1, p2)
def distance_to_line(p0, p1, p2):
    return abs((p2.x-p1.x)*(p1.y-p0.y) - (p1.x-p0.x)*(p2.y-p1.y))/distance(p2,p1)

# Linear equation from 2 points
## Requires two points (p1, p2)
## Returns m, b for y = mx + b
def linear_equation_from_2_points(p1, p2):
    m = (p2.y - p1.y) / (p2.x - p1.x)
    b = p1.y - m*p1.x
    return m, b

# Projection on line
## Requires a point and m,b equation
## Returns the coordinates of the projection on the line
def projection_on_line(p0, m, b, param=-1):
    x = (param * (param * p0.x - m * p0.y) - m * b) / (m * m + param * param)
    y = (m * (-param * p0.x + m * p0.y) - param * b) / (m * m + param * param)
    return x, y


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# PassLine
## Defines a pass lines, wraps functions to handle covered lines
class PassLine:
    def __init__(self, player_a, player_b):
        self.a = player_a
        self.b = player_b

        self.lineblocked = False

    # mad -> maximum allowed distance
    def is_line_blocked(self, player, mad):
        self.lineblocked = False
        if distance_to_line(player, self.a, self.b) <= mad:
            m, b = linear_equation_from_2_points(self.a, self.b)
            x, y = projection_on_line(player, m, b, param=-1)
            if self.a.x <= x <= self.b.x or self.a.x >= x >= self.b.x:
                self.lineblocked = True
                return True
        return False

    def plot(self, color):
        plt.plot([self.a.x, self.b.x], [self.a.y, self.b.y], color=color, alpha=0.5+0.5*self.lineblocked)

    def reset(self):
        self.lineblocked = False

# Team
## Defines a team, wraps the functions to calculate heuristics
### (1) initialize
### (2) add players
### (3) initialize heuristics
### (...)
class Team:
    def __init__(self):
        self.players = []
        self.origin_player = None

        self.passlines = []
        self.goallines = []
    
    def reset_lines(self):
        for line in self.passlines + self.goallines:
            line.reset()
    
    def reset(self):
        self.players = []
        self.passlines = []
        self.goallines = []

    def add_player(self, x, y, origin=False):
        player = Player(x, y)
        self.players.append(player)
        if origin:
           self.origin_player = player
   
    def calculate_random_player_limits(self):
        xxs = [player.x for player in self.players]
        self.x_min = max(16, min(xxs) - 5)
        self.x_max = min(100, max(xxs) + 5)
    
    # mad -> maximum allowed distance
    def initialize_heuristic(self, weights={'w1':1, 'w2':1, 'w3':1, 'w4':4}, mad=5):
        self.w1 = weights['w1']
        self.w2 = weights['w2']
        self.w3 = weights['w3']
        self.w4 = weights['w4']
        self.mad = mad

    # proposed_team = instance of team with players
    def calculate_heuristic(self, proposed_team):
        self.reset_lines()
        c1 = self.calculate_covered_pass_lines(proposed_team)
        c2 = self.calculate_covered_goal_lines(proposed_team)
        c3 = self.calculate_marking_distance(proposed_team)
        c4 = self.calculate_pitch_control(proposed_team)
        c5 = self.validate_positions(proposed_team)

        return self.w1 * c1 + self.w2 * c2 + self.w3 * c3 + self.w4 * c4 + (self.w1 + self.w2 + self.w3 + self.w4) * c5

    def validate_positions(self, proposed_team):
        infractions_counter = 0
        for player in proposed_team.players:
            if (player.x < 15.72) and (20.37 < player.y < 79.63):
                infractions_counter -= 1
        return infractions_counter


    ## Definition:
    ### A pass line is covered if:
    #### A player distance to the line defined by the two players is lower than
    #### & The player X coordinates are between both player's X coordinates
    def calculate_covered_pass_lines(self, proposed_team):
        # If passlines aren't already created, create them
        if self.passlines == []:
            for player in self.players:
                if player != self.origin_player:
                    self.passlines.append(PassLine(self.origin_player, player))

        # Calculate # of lines covered
        covered_passlines_counter = 0
        for def_player in proposed_team.players:
            for passline in self.passlines:
                if passline.lineblocked:
                    continue
                if passline.is_line_blocked(def_player, self.mad):
                    covered_passlines_counter += 1
                    break

        return covered_passlines_counter/10

    def calculate_covered_goal_lines(self, proposed_team):
        # If goallines aren't already created, create them
        imaginary_opp_goal = Player(100, 50)
        if self.goallines == []:
            for player in self.players:
                self.goallines.append(PassLine(player, imaginary_opp_goal))

        # Calculate # of lines covered
        covered_goallines_counter = 0
        for def_player in proposed_team.players:
            for goalline in self.goallines:
                if goalline.lineblocked:
                    continue
                if goalline.is_line_blocked(def_player, self.mad):
                    covered_goallines_counter += 1
                    break

        return covered_goallines_counter/11

    def calculate_marking_distance(self, proposed_team):
        total_marking_distance = 0
        for player in self.players:
            distances = []
            for proposed_team_player in proposed_team.players:
                distances.append(distance(player, proposed_team_player))
            min_distances = min(distances)
            total_marking_distance += min_distances

            # If distance is too close, remove
            if min_distances < 1:
                total_marking_distance += 2


        # Dividing by the distance of a 4-4-2 vs 4-4-2
        ## OFF: (10, 50),(30, 20),(30, 40),(30, 60),(30, 80),(50, 20),(50, 40),(50, 60),(50, 80),(70, 40),(70, 60)
        ## DEF: (90, 50),(80, 20),(80, 40),(80, 60),(80, 80),(60, 20),(60, 40),(60, 60),(60, 80),(40, 40),(40, 60)
        return 1 - total_marking_distance / 156.34
    
    
    def calculate_pitch_control(self, proposed_team):
        attacking_team = [(player.x, player.y, 0) for player in self.players]
        defensive_team = [(player.x, player.y, 1) for player in proposed_team.players]

        df = pd.DataFrame(attacking_team + defensive_team, columns=['x','y','team'])
        df.y /= 120/75 # Normalizing distance along YYs
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(df[['x', 'y']], df.team)
        pred = clf.predict(pd.DataFrame([[i, j/(120/75)] for i in range(101) for j in range(101)], columns=['x','y']))
        pred = pred * np.array([2 if i > (self.x_max-5) else 1 for i in range(101) for _ in range(101)])

        # Dividing by the pitch control of a 4-4-2 vs 4-4-2
        return pred.sum() / 7305

    def plot_team(self, dpi=120):
        draw.pitch(dpi = dpi)
        attacking_team = [(player.x, player.y, 0) for player in self.players]
        for i, player in enumerate(attacking_team):
            plt.scatter(player[0], player[1], s=75, color='blue', edgecolors='navy', linewidths=1,  zorder=10)
            plt.text(player[0], player[1], i, zorder=11, fontsize='xx-small', color='white', ha='center', va='center')

        plt.scatter(attacking_team[0][0]+1, attacking_team[0][1], s=48, color='black',  zorder=11)
        plt.scatter(attacking_team[0][0]+1, attacking_team[0][1], s=48, marker='*', color='white',  zorder=11)
        plt.scatter(attacking_team[0][0]+1, attacking_team[0][1], s=32, marker='+', color='black',  zorder=11)


    def plot_result(self, proposed_team, save_fig_dir='', dpi=120):
        attacking_team = [(player.x, player.y, 0) for player in self.players]
        defensive_team = [(player.x, player.y, 1) for player in proposed_team.players]
        
        draw.pitch(dpi = dpi)
        # Draw initial state
        for i, player in enumerate(attacking_team):
            plt.scatter(player[0], player[1], s=75, color='blue', edgecolors='navy', linewidths=1,  zorder=10)
            plt.text(player[0], player[1], i, zorder=11, fontsize='xx-small', color='white', ha='center', va='center')

        for i, player in enumerate(defensive_team):
            plt.scatter(player[0], player[1], s=75, color='orange', edgecolors='C1', linewidths=1,  zorder=10)
            plt.text(player[0], player[1], i, zorder=11, fontsize='xx-small', color='white', ha='center', va='center')

        # Draw all passing lines
        for passline in self.passlines:
            passline.plot(color='tomato')

        # Draw all goal-defense lines
        for goalline in self.goallines:
            goalline.plot(color='green')

        # Distance between players
        #plt.plot([defensive_team[3][0], attacking_team[8][0]], [defensive_team[3][1], attacking_team[8][1]], color = 'C0')
        #plt.plot([defensive_team[4][0], attacking_team[3][0]], [defensive_team[4][1], attacking_team[3][1]], color = 'C0')
        #plt.plot([defensive_team[9][0], attacking_team[2][0]], [defensive_team[9][1], attacking_team[2][1]], color = 'C0')
        #plt.plot([defensive_team[5][0], attacking_team[7][0]], [defensive_team[5][1], attacking_team[7][1]], color = 'C0')
        #plt.plot([defensive_team[1][0], attacking_team[9][0]], [defensive_team[1][1], attacking_team[9][1]], color = 'C0')
        #plt.plot([defensive_team[2][0], attacking_team[10][0]], [defensive_team[2][1], attacking_team[10][1]], color = 'C0')
        #plt.plot([defensive_team[6][0], attacking_team[6][0]], [defensive_team[6][1], attacking_team[6][1]], color = 'C0')
        #plt.plot([defensive_team[7][0], attacking_team[4][0]], [defensive_team[7][1], attacking_team[4][1]], color = 'C0')
        #plt.plot([defensive_team[8][0], attacking_team[5][0]], [defensive_team[8][1], attacking_team[5][1]], color = 'C0')
        #plt.plot([defensive_team[8][0], attacking_team[1][0]], [defensive_team[8][1], attacking_team[1][1]], color = 'C0')

        # Pitch control
        print_df = pd.DataFrame([[i, j/(120/75)] for i in range(101) for j in range(101)], columns=['x','y'])
        df = pd.DataFrame(attacking_team + defensive_team, columns=['x','y','team'])
        df.y /= 120/75
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(df[['x', 'y']], df.team)
        print_df['team'] = clf.predict(print_df[['x', 'y']])
        print_df.y *= 120/75
        print_df_t1 = print_df[print_df.team == 0]
        plt.scatter(print_df_t1['x'], print_df_t1['y'], 3, c='C0', marker='s', alpha=0.5)
        print_df_t2 = print_df[print_df.team == 1]
        plt.scatter(print_df_t2['x'], print_df_t2['y'], 3, c='C1', marker='s', alpha=0.5)

        if save_fig_dir != '':
            plt.savefig(save_fig_dir, bbox_inches='tight', dpi=dpi)
            plt.clf()
        else:
            plt.show()

        return

        
def create_adversary(opposing_team, weights={'w1':1, 'w2':1, 'w3':1, 'w4':1}):
    adversary = Team()
    adversary.initialize_heuristic(weights=weights, mad=5)
    adversary.add_player(opposing_team[0][0], opposing_team[0][1], True)
    for player in opposing_team[1:]:
        adversary.add_player(player[0], player[1])
    adversary.calculate_random_player_limits()

    return adversary

def create_team(team_coordinates):
    team = Team()
    for player in team_coordinates:
        team.add_player(player[0], player[1]) 
    return team

def generate_random_start(x_min=16, x_max=100, y_min=0, y_max=100):
    # 16 to avoid starting inside the box
    return [[randint(x_min, x_max), randint(y_min, y_max)] for _ in range(10)]

def plot_proposed_team(adversary, current_proposal, save_fig_dir=''):
    proposed_team = create_team(current_proposal)
    adversary.calculate_heuristic(proposed_team)
    adversary.plot_result(proposed_team, save_fig_dir)


opposing_team_1 = [[5, 39], [9, 28], [10, 72], [23, 98], [26, 1], [20, 49], [32, 27], [32, 70], [48, 97], [48, 51], [48, 3]]
default_adversary_1 = create_adversary(opposing_team_1)

opposing_team_2 = [[5, 39], [9, 7], [6, 25], [10, 64], [26, 98], [26, 2], [26, 63], [25, 29], [38, 81], [36, 16], [46, 47]]
default_adversary_2 = create_adversary(opposing_team_2)

opposing_team_3 = [[5, 39], [26, 91], [22, 68], [23, 49], [29, 31], [40, 92], [35, 74], [37, 56], [43, 40], [48, 84], [46, 68]]
default_adversary_3 = create_adversary(opposing_team_3)

opposing_team_4 = [[5, 39], [10, 30], [9, 71], [21, 98], [22, 1], [20, 49], [30, 28], [29, 63], [42, 49], [47, 78], [47, 18]]
default_adversary_4 = create_adversary(opposing_team_4)

opposing_team_5 = [[5, 39], [7, 26], [8, 71], [19, 1], [18, 97], [19, 36], [28, 65], [36, 85], [36, 19], [48, 60], [44, 49]]
default_adversary_5 = create_adversary(opposing_team_5)

formation_442 = [[52, 96], [53, 67], [52, 32], [51, 4], [31, 5], [32, 31], [33, 65], [32, 96], [18, 66], [17, 32]]
default_442_formation = create_adversary(formation_442)