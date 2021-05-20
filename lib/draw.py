import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Arc

from random import random
import numpy as np

#bg_color = '#091442', line_color = '#3562A6'
def pitch(bg_color = '#FFFFFF', line_color = '#000000', dpi = 144):
    # Background cleanup
    plt.rcParams['figure.figsize'] = (10.5,6.8)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.facecolor'] = bg_color
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.scatter(50, 50, s=1000000, marker='s', color=bg_color)

    # Set plotting limit
    plt.xlim([-5, 105])
    plt.ylim([-5, 105])

    # Outside lines
    plt.axvline(0, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    plt.axvline(100, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    plt.axhline(0, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)
    plt.axhline(100, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)

    # Midfield line
    plt.axvline(50, ymin=0.0455, ymax=0.9545, linewidth=1, color=line_color)

    # Goals
    plt.axvline(0, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(100, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(-1, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(101, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)

    # Small Box
    ## (Width-SmallboxWidth)/2/ScaleTo100, (Margin+(Width-SmallboxWidth)/2/ScaleTo100)/(100+Margins)
    ## (68-7.32-11)/2/0.68, (5+((68-7.32-11)/2/.68))/110
    ## (5+5.5/1.05)/110, 5.25/1.05
    plt.axvline(5.24, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)
    plt.axvline(94.76, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)

    plt.axhline(36.53, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)
    plt.axhline(63.47, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)

    plt.axhline(36.53, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)
    plt.axhline(63.47, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)

    # Big Box
    plt.axvline(15.72, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    plt.axhline(20.37, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)
    plt.axhline(79.63, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)

    plt.axvline(84.28, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    plt.axhline(20.37, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color)
    plt.axhline(79.63, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color);

    # Penalty and starting spots and arcs
    plt.scatter([10.4762, 89.5238, 50], [50,50,50], s=1, color=line_color)
    e1 = Arc((10.4762,50), 17.5, 27, theta1=-64, theta2=64, fill=False, color=line_color)
    e2 = Arc((89.5238,50), 17.5, 27, theta1=116, theta2=244, fill=False, color=line_color)
    e3 = Arc((50,50), 17.5, 27, fill=False, color=line_color)
    plt.gcf().gca().add_artist(e1)
    plt.gcf().gca().add_artist(e2)
    plt.gcf().gca().add_artist(e3)


def pitch_for_animation(bg_color = '#091442', line_color = '#3562A6', dpi = 144, figsize=(10.5,6.8)):
    # Background cleanup
    fig = Figure(figsize=figsize, dpi = dpi, facecolor = bg_color)
    a = fig.add_subplot(111)
    a.set_xticks([])
    a.set_yticks([])
    a.set_frame_on(False)
    a.scatter(50, 50, s=1000000, marker='s', color=bg_color)

    # Set plotting limit
    a.set_xlim([-5, 105])
    a.set_ylim([-5, 105])

    # Outside lines
    a.axvline(0, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    a.axvline(100, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    a.axhline(0, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)
    a.axhline(100, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)

    # Midfield line
    a.axvline(50, ymin=0.0455, ymax=0.9545, linewidth=1, color=line_color)

    # Goals
    a.axvline(0, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    a.axvline(100, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    a.axvline(-1, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    a.axvline(101, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)

    # Small Box
    ## (Width-SmallboxWidth)/2/ScaleTo100, (Margin+(Width-SmallboxWidth)/2/ScaleTo100)/(100+Margins)
    ## (68-7.32-11)/2/0.68, (5+((68-7.32-11)/2/.68))/110
    ## (5+5.5/1.05)/110, 5.25/1.05
    a.axvline(5.24, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)
    a.axvline(94.76, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)

    a.axhline(36.53, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)
    a.axhline(63.47, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)

    a.axhline(36.53, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)
    a.axhline(63.47, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)

    # Big Box
    a.axvline(15.72, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    a.axhline(20.37, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)
    a.axhline(79.63, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)

    a.axvline(84.28, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    a.axhline(20.37, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color)
    a.axhline(79.63, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color);

    # Penalty and starting spots and arcs
    a.scatter([10.4762, 89.5238, 50], [50,50,50], s=1, color=line_color)
    e1 = Arc((10.4762,50), 17.5, 27, theta1=-64, theta2=64, fill=False, color=line_color)
    e2 = Arc((89.5238,50), 17.5, 27, theta1=116, theta2=244, fill=False, color=line_color)
    e3 = Arc((50,50), 17.5, 27, fill=False, color=line_color)
    a.add_patch(e1)
    a.add_patch(e2)
    a.add_patch(e3)

    return fig, a


def clear_objects(objects):
    for obj in objects:
        obj.remove()

def random_checkpoints(ax, no_checkpoints, width, no_angle=True):
    checkpoint_positions = []
    checkpoint_positions_list = []
    checkpoint_objects = []

    for _ in range(no_checkpoints):
        mid_point_x = random() * 100
        mid_point_y = random() * 100

        if no_angle:
            angle = np.pi/2
        else:
            angle = random() * 2 * np.pi

        p1_x = mid_point_x + width * np.cos(angle)
        p1_y = mid_point_y + width * np.sin(angle) * 10.5/6.8
        p2_x = mid_point_x - width * np.cos(angle)
        p2_y = mid_point_y - width * np.sin(angle) * 10.5/6.8

        checkpoint_objects.append(ax.scatter([p1_x, p2_x], [p1_y, p2_y], s=100, marker = 'h', color = 'orange', zorder = 5, linewidths = 2, edgecolors = 'white'))
        checkpoint_objects.append(ax.plot([p1_x, p2_x], [p1_y, p2_y], color = 'yellow', zorder = 4)[0])

        checkpoint_positions_list.append([p1_x, p1_y, p2_x, p2_y])
        checkpoint_positions += [p1_x, p1_y, p2_x, p2_y]
    
    return checkpoint_positions, checkpoint_positions_list, checkpoint_objects

def fixed_checkpoints(ax, no_checkpoints, width, no_angle=True):
    checkpoint_positions = []
    checkpoint_positions_list = []
    checkpoint_objects = []

    for i in range(no_checkpoints):
        mid_point_x = 100 / (no_checkpoints + 1) + i * 100 / (no_checkpoints + 1)
        mid_point_y = 40 + 20 * (i % 2)

        if no_angle:
            angle = np.pi/2
        else:
            angle = random() * 2 * np.pi

        p1_x = mid_point_x + width * np.cos(angle)
        p1_y = mid_point_y + width * np.sin(angle) * 10.5/6.8
        p2_x = mid_point_x - width * np.cos(angle)
        p2_y = mid_point_y - width * np.sin(angle) * 10.5/6.8

        checkpoint_objects.append(ax.scatter([p1_x, p2_x], [p1_y, p2_y], s=100, marker = 'h', color = 'orange', zorder = 5, linewidths = 2, edgecolors = 'white'))
        checkpoint_objects.append(ax.plot([p1_x, p2_x], [p1_y, p2_y], color = 'yellow', zorder = 4)[0])

        checkpoint_positions_list.append([p1_x, p1_y, p2_x, p2_y])
        checkpoint_positions += [p1_x, p1_y, p2_x, p2_y]
    
    return checkpoint_positions, checkpoint_positions_list, checkpoint_objects


def plot_pass(x, y, endX, endY, success=True, alpha=1):
    if success == True:
        plt.arrow(x, y, endX - x, endY - y, color='#611a6a', alpha=alpha, width = 0.5, label='Pass', length_includes_head=True, zorder = 22) #purple
    else:
        plt.arrow(x, y, endX - x, endY - y, color='#611a05', alpha=alpha) #brown


def plot_shot(x, y, success=True, alpha=1):
    if success == True:
        plt.arrow(x, y, 100 - x, 50 - y, color='#61b26a', alpha=alpha, width = 0.5, label='Shot', length_includes_head=True, zorder = 20) #green
    else:
        plt.arrow(x, y, 100 - x, 50 - y, color='#c11a6a', alpha=alpha) #pink

def plot_dribble(x, y, endX, endY, success=True, alpha=1):
    if success == True:
        plt.arrow(x, y, endX - x, endY - y, color='#ffe200', alpha=alpha, width = 0.5, label='Dribble', length_includes_head=True, zorder = 21) #yellow
    else:
        plt.arrow(x, y, endX - x, endY - y, color='#ff0000', alpha=alpha) #red

def plot_rebound(x, y, alpha=1):
    plt.scatter(x, y, marker='x', color='#000000', alpha=alpha)



