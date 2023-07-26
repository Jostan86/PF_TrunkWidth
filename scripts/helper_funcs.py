#!/usr/bin/env python3

import json
import matplotlib
matplotlib.use("TkAgg")  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import time
def get_map_data(include_sprinklers=False, separate_test_trees=False, move_origin=True, origin_offset=5):
    # Path to the tree data dictionary
    tree_data_path = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data' \
                     '/2020_11_bag_data/afternoon2/tree_list_mod3.json'

    # Load the tree data dictionary
    with open(tree_data_path, 'rb') as f:
        tree_data = json.load(f)

    # Extract the classes and positions from the tree data dictionary
    classes = []
    positions = []
    widths = []
    for tree in tree_data:



        # Skip sprinklers if include_sprinklers is False
        if tree['class_estimate'] == 2 and not include_sprinklers:
            continue

        # Save test trees as a different class if separate_test_trees is True
        if tree['test_tree'] and separate_test_trees:
            classes.append(3)
        else:
            classes.append(tree['class_estimate'])

        # Move the tree positions by the gps adjustment
        position_estimate = np.array(tree['position_estimate']) + np.array(tree['gps_adjustment'])
        positions.append(position_estimate.tolist())
        widths.append(tree['width_estimate'])

    # Make the classes and positions numpy arrays
    classes = np.array(classes, dtype=int)
    positions = np.array(positions)
    widths = np.array(widths)

    # Remove the sprinklers
    positions = positions[classes != 2]
    widths = widths[classes != 2]
    classes = classes[classes != 2]

    if move_origin:
        # Find the min x and y values, subtract 5 and set that as the origin
        x_min = np.min(positions[:, 0]) - origin_offset
        y_min = np.min(positions[:, 1]) - origin_offset

        # Subtract origin from positions
        positions[:, 0] -= x_min
        positions[:, 1] -= y_min

    return classes, positions, widths

class ParticleMapPlotter:
    def __init__(self, map_data):
        plt.ion()

        # Set the figure size and dpi
        plt.figure(figsize=(20, 20))

        # Set text size
        plt.rcParams.update({'font.size': 22})

        classes = map_data['classes']
        tree_positions = map_data['positions'][classes == 0]
        post_positions = map_data['positions'][classes == 1]

        plt.scatter(tree_positions[:, 0], tree_positions[:, 1], s=30, c='g', label='Trees')
        plt.scatter(post_positions[:, 0], post_positions[:, 1], s=30, c='r', label='Posts')
        plt.scatter([7], [43], s=30, c='b', label='start')

        # self.dynamic_plot = plt.plot(particles[:, 0], particles[:, 1], 'b.', markersize=3, label='Particles')[0]

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.title('Particle Filter Visualization')

        # # Set axis limits
        # plt.xlim([0, 30])
        # plt.ylim([25, 65])

        # Set the plot to be equal aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')

        # Set the plot to show gridlines
        plt.grid(True)

        # Register the click event handler
        cid = plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)

        plt.show()
        plt.pause(0.01)

    def add_particle(self, particles):
        self.dynamic_plot = plt.plot(particles[:, 0], particles[:, 1], 'b.', markersize=3, label='Particles')[0]
        plt.pause(0.01)


    def update_particles(self, particles):
        # start_time = time.time()
        self.particles = particles
        self.dynamic_plot.set_data(particles[:, 0], particles[:, 1])
        plt.pause(0.01)
        # print("plot update time: ", time.time() - start_time)

    def on_click(self, event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        # ans = input("Place circle at ({}, {})? (y/n): ".format(round(x, 2), round(y, 2)))
        # if ans == 'y':
        #     print(f"Circle placed at ({x}, {y})")
        #     plt.gca().add_patch(plt.Circle((x, y), 2, color='r', fill=True))
        #     plt.draw()
        print("clicked at ({}, {})".format(round(x, 2), round(y, 2)))