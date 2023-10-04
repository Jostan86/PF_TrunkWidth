#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
import os
import rosbag
from width_estimation import TrunkAnalyzer
import json
import numpy as np
from pf_engine_cpy import PFEngine
import cv2
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QSlider, QComboBox, QSizePolicy, QPlainTextEdit, QCheckBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
import bisect
import time
import pyqtgraph as pg
import pandas as pd
import csv
from scipy.ndimage import label
from scipy.spatial import KDTree
import math

# To run in pycharm add the following environment variable:
# LD_PRELOAD: Set it to /usr/lib/x86_64-linux-gnu/libstdc++.so.6

def get_map_data(include_sprinklers=False, move_origin=True, origin_offset=5):
    # Path to the tree data dictionary
    tree_data_path = '/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data' \
                     '/2020_11_bag_data/afternoon2/tree_list_mod4.json'
    # tree_data_path = '/media/jostan/portabits/sept6/tree_list_mod4.json'

    # Load the tree data dictionary
    with open(tree_data_path, 'rb') as f:
        tree_data = json.load(f)

    # Extract the classes and positions from the tree data dictionary
    classes = []
    positions = []
    widths = []
    tree_nums = []
    test_tree_nums = []

    for tree in tree_data:

        # Skip sprinklers if include_sprinklers is False
        if tree['class_estimate'] == 2 and not include_sprinklers:
            continue

        # Save test trees as a different class if separate_test_trees is True
        if tree['test_tree']:
            test_tree_nums.append(int(tree['test_tree_num']))
        else:
            test_tree_nums.append(None)

        classes.append(tree['class_estimate'])
        # Move the tree positions by the gps adjustment
        position_estimate = np.array(tree['position_estimate']) + np.array(tree['gps_adjustment'])
        positions.append(position_estimate.tolist())
        widths.append(tree['width_estimate'])
        tree_nums.append(tree['tree_num'])

    # Make the classes and positions numpy arrays
    classes = np.array(classes, dtype=int)
    positions = np.array(positions)
    widths = np.array(widths)

    if move_origin:
        # Find the min x and y values, subtract 5 and set that as the origin
        x_min = np.min(positions[:, 0]) - origin_offset
        y_min = np.min(positions[:, 1]) - origin_offset

        # Subtract origin from positions
        positions[:, 0] -= x_min
        positions[:, 1] -= y_min

    map_data = {'classes': classes, 'positions': positions, 'widths': widths, 'tree_nums': tree_nums, 'test_tree_nums': test_tree_nums}

    return map_data

def get_test_starts():
    df = pd.read_csv("../data/test_starts.csv", header=1)

    starts = []

    for i in range(len(df['x'])):

        # Tests start 1
        starts.append({"run_num": df['run'][i],
                  "start_time": df['time'][i],
                  "start_pose_center": [df['x'][i], df['y'][i]],
                  "start_pose_width": df['width'][i],
                  "start_pose_height": df['height'][i],
                  "start_pose_rotation": df['rotation'][i],})
    return starts


class ParticleMapPlotter(QMainWindow):
    def __init__(self, map_data):
        super().__init__()

        self.show_nums = True

        # Create a pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()

        self.plot_widget.setAspectLocked(True, ratio=1)
        self.plot_widget.setBackground('w')  # 'w' is short for white

        # Configure the main window and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout()
        self.main_widget.setLayout(layout)
        layout.addWidget(self.plot_widget)

        # Set the map data
        self.positions = np.array(map_data['positions'])
        self.classes = np.array(map_data['classes'])
        self.tree_numbers = map_data['tree_nums']
        self.test_tree_nums = map_data['test_tree_nums']
        test_trees = np.array([0 if num is None else 1 for num in map_data['test_tree_nums']])
        # Set class to 3 for test trees
        self.classes[test_trees == 1] = 3

        # # Draw the map
        self.draw_plot()

    def draw_plot(self, particles=None):
        """Method to draw the map on the plot widget"""

        # row_num_xs = [4.9, 5.5, 6.05, 6.65, 7.4, 7.45, 7.9, 8.65, 9.2, 9.65, 10.25, 10.65, 11.05, 11.6, 12.1, 12.65,
        #               13.2]
        # row_num_ys = [4.9, 10.8, 16.5, 22.35, 28.1, 33.3, 39.05, 45.05, 51.05, 56.5, 62.6, 68.25, 73.9, 79.55, 85.6,
        #               91.5, 97.3]
        # row_nums = [i for i in range(len(row_num_xs))]

        self.plot_widget.clear()

        tree_positions = self.positions[self.classes == 0]
        post_positions = self.positions[self.classes == 1]
        test_tree_positions = self.positions[self.classes == 3]


        self.dot_size = 10

        # Adding data to the plot widget
        self.plot_widget.plot(tree_positions[:, 0], tree_positions[:, 1], pen=None, symbol='o', symbolBrush='g',
                              symbolSize=self.dot_size, name='Trees')
        self.plot_widget.plot(post_positions[:, 0], post_positions[:, 1], pen=None, symbol='o', symbolBrush='r',
                              symbolSize=self.dot_size, name='Posts')
        self.plot_widget.plot(test_tree_positions[:, 0], test_tree_positions[:, 1], pen=None, symbol='o',
                              symbolBrush=(0, 81, 180), symbolSize=self.dot_size, name='Test Trees')

        # Add numbers to the trees if show_nums is True, which is toggled by a button in the app
        if self.show_nums:
            for i, (x, y) in enumerate(self.positions):
                # tree_num_text = pg.TextItem(text=str(self.tree_numbers[i]), color=(0, 0, 0), anchor=(0.5, 0.5))
                tree_num_text = pg.TextItem(
                    html='<div style="text-align: center"><span style="color: #000000; font-size: 8pt;">{}</span></div>'.format(
                            self.tree_numbers[i]), anchor=(1.1, 0.5))
                tree_num_text.setPos(x, y)
                self.plot_widget.addItem(tree_num_text)

                # Add test tree numbers
                if self.classes[i] == 3:
                    tree_num_text = pg.TextItem(
                        html = '<div style="text-align: center"><span style="color: #000000; font-size: 8pt;">{}</span></div>'.format(
                            self.test_tree_nums[i]), anchor = (-0.1, 0.5))
                    tree_num_text.setPos(x, y)
                    self.plot_widget.addItem(tree_num_text)

        # Add particles to the plot widget
        # If no particles are given, add 100 particles at the origin, as a placeholder. Otherwise, add the given particles
        if particles is None:
            particles = np.zeros((100, 2))

        self.particle_plot_item = self.plot_widget.plot(particles[:, 0], particles[:, 1], pen=None, symbol='o',
                                                        symbolBrush='b', symbolSize=2, name='Particles')

        self.best_guess_plot_item = self.plot_widget.plot([], [], pen=None, symbol='o', symbolBrush='r',
                                                          symbolSize=10, name='Test Trees')

        self.in_progress_tree_plot_item = self.plot_widget.plot([], [], pen=None, symbol='o', symbolBrush=(211, 0,
                                                                                                           255),
                                                          symbolSize=self.dot_size, name='Test Trees')

        self.complete_plot_item = self.plot_widget.plot([], [], pen=None, symbol='o', symbolBrush=(138, 187, 248),
                                                          symbolSize=self.dot_size, name='Test Trees')



        if particles is not None:
            self.update_particles(particles)

    def update_particles(self, particles):

        if particles is not None:
            particles = self.downsample_particles(particles)
            self.particle_plot_item.setData(particles[:, 0], particles[:, 1])
        else:
            particles = np.zeros((100, 2))
            self.particle_plot_item.setData(particles[:, 0], particles[:, 1])


    def update_best_guess(self, best_guess_position):
        if best_guess_position is not None:
            self.best_guess_plot_item.setData([best_guess_position[0]], [best_guess_position[1]])
        else:
            self.best_guess_plot_item.setData([], [])

    def update_in_progress_tree(self, in_progress_tree_position):
        if in_progress_tree_position is not None:
            self.in_progress_tree_plot_item.setData([in_progress_tree_position[0]], [in_progress_tree_position[1]])
        else:
            self.in_progress_tree_plot_item.setData([], [])

    def update_complete(self, complete_position):

        if complete_position is None:
            self.complete_plot_item.setData([], [])
        elif len(complete_position) == 1:
            complete_position = complete_position[0]
            self.complete_plot_item.setData([complete_position[0]], [complete_position[1]])
        else:
            self.complete_plot_item.setData(complete_position[:, 0], complete_position[:, 1])

    def downsample_particles(self, particles, max_samples=10000):
        """
        Downsample a 2D array of particles to a maximum number of samples. This is useful for plotting large numbers of
        particles without slowing down the GUI.

        Parameters:
        - particles: 2D numpy array of shape (n, 2)
        - max_samples: int, maximum number of samples after downsampling

        Returns:
        - Downsampled 2D numpy array of particles
        """
        num_particles = particles.shape[0]
        if num_particles <= max_samples:
            return particles

        indices = np.random.choice(num_particles, max_samples, replace=False)
        return particles[indices]

class MyMainWindow(QMainWindow):
    def __init__(self):
        """Constructs the GUI for the particle filter app"""
        super(MyMainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle("My PyQt App")
        self.setGeometry(0, 0, 1900, 1050)

        desktop = QApplication.desktop()
        target_screen_number = 0
        if target_screen_number < desktop.screenCount():
            target_screen = desktop.screen(target_screen_number)
            self.move(target_screen.geometry().left(), target_screen.geometry().top())

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Left-side widgets
        self.reset_button = QPushButton("Reset")

        # Setup start location layout
        self.start_x_input = QLineEdit()
        self.start_y_input = QLineEdit()
        start_location_layout = QHBoxLayout()
        start_location_layout.addWidget(QLabel("Start Pose Center:"))
        start_location_layout.addWidget(QLabel("X:"))
        start_location_layout.addWidget(self.start_x_input)
        start_location_layout.addWidget(QLabel("Y:"))
        start_location_layout.addWidget(self.start_y_input)

        # Setup width and height layout
        self.width_input = QLineEdit()
        self.height_input = QLineEdit()
        self.rotation_input = QLineEdit()
        width_height_layout = QHBoxLayout()
        width_height_layout.addWidget(QLabel("Width:"))
        width_height_layout.addWidget(self.width_input)
        width_height_layout.addWidget(QLabel("Height:"))
        width_height_layout.addWidget(self.height_input)
        width_height_layout.addWidget(QLabel("Rotation (deg):"))
        width_height_layout.addWidget(self.rotation_input)

        # Setup num particles layout
        self.particle_density_input = QLineEdit()
        num_particles_layout = QHBoxLayout()
        self.particle_density_label = QLabel("Particle Density:")
        num_particles_layout.addWidget(self.particle_density_label)

        num_particles_layout.addWidget(self.particle_density_input)
        num_particles_layout.addWidget(QLabel("particles/m^2"))
        # Add fixed width space
        num_particles_layout.addSpacing(50)
        num_particles_layout.addWidget(QLabel("Num Particles:"))
        self.num_particles_label = QLabel("0")
        num_particles_layout.addWidget(self.num_particles_label)
        num_particles_layout.addStretch(1)


        # Setup checkboxes
        self.include_width_checkbox = QCheckBox("Include width in weight calculation")
        self.include_width_checkbox.setChecked(True)
        self.save_data_checkbox = QCheckBox("Save data")
        self.save_data_checkbox.setChecked(False)
        self.stop_when_converged = QCheckBox("Stop When Converged")
        self.stop_when_converged.setChecked(False)
        self.signal_test_trees_checkbox = QCheckBox("Signal Test Trees")
        self.signal_test_trees_checkbox.setChecked(True)

        checkbox_layout1 = QHBoxLayout()
        checkbox_layout1.addWidget(self.include_width_checkbox)
        checkbox_layout1.addWidget(self.stop_when_converged)
        checkbox_layout1.addWidget(self.save_data_checkbox)
        checkbox_layout1.addWidget(self.signal_test_trees_checkbox)
        checkbox_layout1.addStretch(1)

        self.use_saved_data_checkbox = QCheckBox("Use saved data")
        self.use_saved_data_checkbox.setChecked(False)
        self.show_image_checkbox = QCheckBox("Show image")
        self.show_image_checkbox.setChecked(True)
        self.update_plot_checkbox = QCheckBox("Update plot")
        self.update_plot_checkbox.setChecked(True)

        checkbox_layout2 = QHBoxLayout()
        checkbox_layout2.addWidget(self.use_saved_data_checkbox)
        checkbox_layout2.addWidget(self.show_image_checkbox)
        checkbox_layout2.addWidget(self.update_plot_checkbox)
        checkbox_layout2.addStretch(1)


        # Add a selection for the mode
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Scroll Images")
        self.mode_selector.addItem("Continuous")
        self.mode_selector.addItem("Manual - single step")
        self.mode_selector.addItem("Manual - single image")
        self.mode_selector.addItem("Tests")
        self.mode_selector.setFixedWidth(300)

        mode_selector_layout = QHBoxLayout()
        # Add label with set width
        mode_label = QLabel("Mode:")
        mode_selector_layout.addWidget(mode_label)
        mode_selector_layout.addWidget(self.mode_selector)
        mode_selector_layout.addStretch(1)

        # Add a selection for the test to run
        self.test_selector = QComboBox()
        self.test_selector.addItem("Test 1")
        self.test_selector.addItem("Test 2")
        self.test_selector.addItem("Test 3")
        self.test_selector.addItem("Test 4")
        self.test_selector.addItem("Test 5")
        self.test_selector.setFixedWidth(200)
        test_selector_layout = QHBoxLayout()
        test_selector_layout.addWidget(QLabel("Test:"))
        test_selector_layout.addWidget(self.test_selector)
        test_selector_layout.addStretch(1)

        self.continue_button = QPushButton("Continue")
        self.start_stop_button = QPushButton("Start")
        self.start_continue_layout = QHBoxLayout()
        self.start_continue_layout.addWidget(self.start_stop_button)
        self.start_continue_layout.addWidget(self.continue_button)

        self.picture_label = QLabel(self)
        self.picture_label.resize(640, 480)
        self.picture_label.setAlignment(Qt.AlignCenter)

        self.img_number_label = QLabel(self)
        self.img_number_label.setAlignment(Qt.AlignCenter)

        # Setup img browsing layouts
        img_browsing_buttons_layout = QHBoxLayout()
        self.prev_img_button = QPushButton("Previous")
        self.next_img_button = QPushButton("Next")
        self.play_fwd_button = QPushButton("Play")
        img_browsing_buttons_layout.addWidget(self.prev_img_button)
        img_browsing_buttons_layout.addWidget(self.next_img_button)
        img_browsing_buttons_layout.addWidget(self.play_fwd_button)

        # Setup bag file selector
        self.data_file_selector = QComboBox()
        data_file_selector_layout = QHBoxLayout()
        data_file_selector_label = QLabel("Current Bag File:")
        data_file_selector_label.setFixedWidth(220)
        self.data_file_open_button = QPushButton("Open")
        self.data_file_open_button.setFixedWidth(120)
        data_file_selector_layout.addWidget(data_file_selector_label)
        data_file_selector_layout.addWidget(self.data_file_selector)
        data_file_selector_layout.addWidget(self.data_file_open_button)

        # Setup bag time line
        self.bag_time_line = QLineEdit()
        bag_time_line_layout = QHBoxLayout()
        bag_time_line_layout.addWidget(QLabel("Current Bag Time:"))
        bag_time_line_layout.addWidget(self.bag_time_line)

        # Add an area to write messages
        self.console = QPlainTextEdit(self)
        self.console.setReadOnly(True)

        # Add a button to clear the console
        self.clear_console_button = QPushButton("Clear Console")

        # Add a button to remove the nums from the plot
        self.plot_nums_toggle_button = QPushButton("Remove Nums")

        # Add a save data button
        self.save_data_button = QPushButton("Save Data")

        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.addWidget(self.clear_console_button)
        bottom_buttons_layout.addWidget(self.plot_nums_toggle_button)
        bottom_buttons_layout.addWidget(self.save_data_button)

        save_diam_data_layout = QHBoxLayout()
        self.tree_num_line = QLineEdit()
        self.tree_num_line.setFixedWidth(100)
        tree_num_label = QLabel("Test Tree Number:")
        tree_num_label.setFixedWidth(150)
        save_diam_data_layout.addWidget(tree_num_label)
        save_diam_data_layout.addWidget(self.tree_num_line)
        self.save_diam_data_button = QPushButton("Begin Saving")
        save_diam_data_layout.addWidget(self.save_diam_data_button)

        self.map_data = get_map_data()

        self.plotter = ParticleMapPlotter(self.map_data)

        # Set up layouts
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.reset_button)
        left_layout.addLayout(start_location_layout)
        left_layout.addLayout(width_height_layout)
        left_layout.addLayout(num_particles_layout)
        left_layout.addLayout(checkbox_layout1)
        left_layout.addLayout(checkbox_layout2)
        left_layout.addLayout(mode_selector_layout)
        left_layout.addLayout(test_selector_layout)
        left_layout.addLayout(self.start_continue_layout)
        left_layout.addWidget(self.picture_label)
        left_layout.addWidget(self.img_number_label)
        left_layout.addLayout(img_browsing_buttons_layout)
        left_layout.addLayout(data_file_selector_layout)
        left_layout.addLayout(bag_time_line_layout)
        left_layout.addWidget(self.console)
        left_layout.addLayout(bottom_buttons_layout)
        left_layout.addLayout(save_diam_data_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plotter)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        central_widget.setLayout(main_layout)

        # Connect the clear console button
        self.clear_console_button.clicked.connect(self.clear_console)

        self.load_image()

    def clear_console(self):
        self.console.clear()

    def load_image(self, img=None):
        """
        Load an image into the GUI image viewer
        Parameters
        ----------
        img : OpenCV image

        Returns None
        -------

        """
        # If image is none make a blank image
        if img is None:
            img = np.ones((480, 640, 3), dtype=np.uint8) * 155

        # Convert the image to a Qt image and display it
        image_cv2 = img
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        image_qt = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image_qt)
        pixmap_scaled = pixmap.scaled(self.picture_label.size(), Qt.KeepAspectRatio)
        self.picture_label.setPixmap(pixmap_scaled)

        # update to ensure the image is shown
        QApplication.processEvents()

class ParticleFilterBagFiles:
    def __init__(self, app):
        self.bridge = CvBridge()

        self.qt_window = MyMainWindow()
        self.qt_app = app

        self.initialize_startup_params1()

        self.img_base_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/yolo_segs/"
        self.saved_data_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/pf_data/"

        self.data_file_names = []

        self.msg_order = []
        self.paired_imgs = []
        self.odom_msgs = []
        self.time_stamps = []
        self.cur_data_pos = 0
        self.cur_odom_pos = 0
        self.cur_img_pos = 0
        self.cur_bag_file_name = None

        self.start_time = None

        self.pf_active = False

        self.converged_once = False

        self.qt_window.start_x_input.setText(str(self.start_pose_center[0]))
        self.qt_window.start_y_input.setText(str(self.start_pose_center[1]))
        self.qt_window.width_input.setText(str(self.start_width))
        self.qt_window.height_input.setText(str(self.start_height))
        self.qt_window.rotation_input.setText(str(self.start_rotation))
        self.qt_window.particle_density_input.setText(str(self.particle_density))

        self.map_data = get_map_data()
        self.test_tree_nums = np.array([0 if num is None else num for num in self.map_data['test_tree_nums']])
        self.test_tree_positions = self.map_data['positions'][self.test_tree_nums > 0]
        self.test_tree_nums = self.test_tree_nums[self.test_tree_nums > 0]
        self.treatment_status = np.zeros(len(self.test_tree_nums))
        self.test_kdtree = KDTree(self.test_tree_positions)


        self.trunk_analyzer = TrunkAnalyzer()


        self.qt_window.reset_button.clicked.connect(self.reset_app)
        self.qt_window.start_stop_button.clicked.connect(self.start_stop_button_clicked)
        self.qt_window.data_file_open_button.clicked.connect(self.open_bag_file_button_clicked)
        self.qt_window.continue_button.clicked.connect(self.cont_button_clicked)
        self.qt_window.continue_button.setEnabled(False)
        self.qt_window.prev_img_button.clicked.connect(self.prev_button_clicked)
        self.qt_window.next_img_button.clicked.connect(self.next_button_clicked)
        self.qt_window.play_fwd_button.clicked.connect(self.play_button_clicked)
        self.qt_window.plot_nums_toggle_button.clicked.connect(self.toggle_plot_nums)
        self.qt_window.save_data_button.clicked.connect(self.save_data_button_clicked)

        self.qt_window.bag_time_line.returnPressed.connect(self.time_stamp_line_edited)

        self.qt_window.include_width_checkbox.stateChanged.connect(self.include_width_changed)
        self.qt_window.use_saved_data_checkbox.stateChanged.connect(self.use_saved_data_changed)
        self.qt_window.show_image_checkbox.stateChanged.connect(self.show_image_changed)
        self.qt_window.update_plot_checkbox.stateChanged.connect(self.show_plot_changed)
        self.qt_window.save_data_checkbox.stateChanged.connect(self.save_data_checkbox_changed)

        # Connect the mode selector
        self.qt_window.mode_selector.currentIndexChanged.connect(self.mode_changed)
        # self.qt_window.test_selector.currentIndexChanged.connect(self.test_changed)

        self.qt_window.save_diam_data_button.clicked.connect(self.save_diam_data_button_clicked)

        self.save_diam_data = False
        self.width_estimates = []
        self.img_x_positions = []
        self.save_diam_data_dir = "../data/diam_data/"
        self.diam_data_set_name = "mope_2023_"


        self.use_loaded_data = False
        self.round_exclude_time = 0

        self.test_starts = get_test_starts()
        self.test_start_num = 0
        self.num_trials = 3
        self.distances = []
        self.convergences = []
        self.run_times = []

        self.save_data = True
        self.saved_data = {}
        self.time_stamps_img = []

        self.time_stamps_keys = []
        self.cur_data_pos = 0

        self.img_data = None
        self.odom_data = None
        self.show_loaded_img = self.qt_window.show_image_checkbox.isChecked()
        self.show_plot_updates = self.qt_window.update_plot_checkbox.isChecked()

        self.run_num = 0
        self.convergence_threshold = 0.5

        # self.qt_window.use_saved_data_checkbox.setChecked(True)
        self.qt_window.stop_when_converged.setChecked(True)

        self.reset_app()
        self.mode_changed()
        self.use_saved_data_changed()

        # Select the first item in the combo box
        if self.starting_bag_index is not None:
            self.qt_window.data_file_selector.setCurrentIndex(self.starting_bag_index)
            self.open_bag_file(self.qt_window.data_file_selector.currentText())
        if self.starting_time is not None:
            self.qt_window.bag_time_line.setText(str(self.starting_time))
            self.time_stamp_line_edited()


        # self.qt_window.mode_selector.setCurrentIndex(1)
        # self.mode_changed()


    def initialize_startup_params1(self):

        self.bag_file_dir = "/media/jostan/MOAD/research_data/achyut_data/sept6/"
        # self.bag_file_dir = "/media/jostan/portabits/sept6/"
        self.topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", "/odometry/filtered"]

        # Start for 72 sec in row106_107_sept.bag, at tree 795
        self.start_pose_center = [28, 95]
        self.particle_density = 250
        self.start_width = 20
        self.start_height = 20
        self.start_rotation = 0

        # Start for beginning of row106_107_sept.bag
        # self.start_pose_center = [11, 66.3]
        # self.particle_density = 500
        # self.start_width = 2
        # self.start_height = 10
        # self.start_rotation = -32

        self.x_offset = 0.8
        self.y_offset = -0.55

        self.starting_time = 72
        self.starting_bag_index = 1

    def initialize_startup_params2(self):

        self.bag_file_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/pcl_mod/"
        self.topics = ["/registered/rgb/image", "/registered/depth/image", "/odometry/filtered"]

        self.start_pose_center = [16, 99.2]
        self.particle_density = 500
        self.start_width = 6
        self.start_height = 22
        self.start_rotation = -32
        self.starting_time = 27.2
        self.starting_bag_index = 67

        # Whole map for when you're feeling arrogant
        # self.start_pose_center = [20, 70]
        # self.particle_density = 500
        # self.start_width = 60
        # self.start_height = 100
        # self.start_rotation = -32


        # self.x_offset = 0
        # self.y_offset = 0
        self.x_offset = 0.8
        self.y_offset = -0.55


    def reset_app(self):
        self.saved_data = {}
        start_x = float(self.qt_window.start_x_input.text())
        start_y = float(self.qt_window.start_y_input.text())
        self.start_pose_center = [start_x, start_y]
        self.start_height = float(self.qt_window.height_input.text())
        self.start_width = float(self.qt_window.width_input.text())
        self.start_rotation = float(self.qt_window.rotation_input.text())
        self.particle_density = int(self.qt_window.particle_density_input.text())
        self.num_particles = int(self.particle_density * self.start_height * self.start_width)
        self.qt_window.num_particles_label.setText(str(self.num_particles))

        self.converged_once = False

        self.pf_engine = PFEngine(self.map_data, start_pose_center=self.start_pose_center,
                                  start_pose_height=self.start_height, start_pose_width=self.start_width,
                                  num_particles=self.num_particles, rotation=np.deg2rad(self.start_rotation),)
        self.pf_engine.include_width = self.qt_window.include_width_checkbox.isChecked()

        # # Good settings for achyut's data:
        # self.pf_engine.R = np.diag([.6, np.deg2rad(20.0)]) ** 2
        # self.pf_engine.dist_sd = 0.4
        # self.pf_engine.width_sd = 0.03
        #
        # self.pf_engine.bin_size = 0.5
        # self.pf_engine.bin_angle = np.deg2rad(4.0)
        # self.pf_engine.epsilon = 0.03

        # Good settings for feb_2023 data:
        self.pf_engine.R = np.diag([.7, np.deg2rad(25.0)]) ** 2
        self.pf_engine.dist_sd = 0.45
        self.pf_engine.width_sd = 0.03

        self.pf_engine.bin_size = 0.8
        self.pf_engine.bin_angle = np.deg2rad(6.0)
        self.pf_engine.epsilon = 0.035

        self.treatment_status = np.zeros(len(self.test_tree_nums))

        self.qt_window.plotter.update_best_guess(None)
        self.qt_window.plotter.update_in_progress_tree(None)
        self.qt_window.plotter.update_complete(None)

        self.qt_window.plotter.update_particles(self.pf_engine.particles)
        # self.qt_window.plotter.draw_plot(self.pf_engine.particles)

        self.first_odom = True
        self.odom_counter = 0



    def start_stop_button_clicked(self):
        # Check current text on button
        if self.qt_window.start_stop_button.text() == "Start":
            self.qt_window.start_stop_button.setText("Stop")


            round_start_time = time.time()

            if self.qt_window.mode_selector.currentText() == "Continuous":
                self.pf_active = True
                self.run_pf()
                self.post_a_time(round_start_time, "Total time:", ms=False)

                if self.use_loaded_data:
                    correct_convergence, distance = self.check_converged_location()
                    print("Correct convergence: {}".format(correct_convergence))
                    print("Distance: {}".format(distance))

            elif self.qt_window.mode_selector.currentText() == "Tests":
                self.qt_window.console.appendPlainText("Starting tests...")
                self.qt_app.processEvents()

                for _ in range(len(self.test_starts)):
                    run_times = []
                    trials_converged = []
                    distances = []

                    for _ in range(self.num_trials):

                        self.qt_window.use_saved_data_checkbox.setChecked(True)
                        self.qt_window.stop_when_converged.setChecked(True)
                        self.qt_window.save_data_checkbox.setChecked(False)
                        self.qt_app.processEvents()

                        # while True:
                        round_start_time = time.time()
                        self.round_exclude_time = 0

                        self.pf_active = True
                        self.run_pf()

                        if self.qt_window.start_stop_button.text() == "Start":
                            break

                        run_times.append(time.time() - round_start_time - self.round_exclude_time)
                        correct_convergence, distance = self.check_converged_location()
                        distances.append(distance)
                        trials_converged.append(correct_convergence)
                        self.load_next_trial()

                    if self.qt_window.start_stop_button.text() == "Start":
                        break

                    # print run times to console
                    self.qt_window.console.appendPlainText("Results for start location {}: ".format(self.test_start_num))
                    for runtime, distance, convergence in zip(run_times, distances, trials_converged):
                        msg = str(round(runtime, 2)) + "s" + "   Converged: " + str(convergence) + "   Distance: " + \
                              str(round(distance, 2)) + "m"

                        self.qt_window.console.appendPlainText(msg)

                    self.distances.append(distances)
                    self.convergences.append(trials_converged)
                    self.run_times.append(run_times)

                    self.qt_app.processEvents()
                    self.test_start_num += 1

                self.process_results()
                self.start_stop_button_clicked()


        else:
            self.qt_window.start_stop_button.setText("Start")
            self.pf_active = False

    def process_results(self):
        convergences = np.array(self.convergences)
        run_times = np.array(self.run_times)

        # Calculate the convergence rate for each start location
        convergence_rates = np.sum(convergences, axis=1) / self.num_trials

        # Calculate the average time to convergence for each start location
        avg_times = np.sum(run_times, axis=1) / self.num_trials

        # Calculate the average time for trials that converged for each start location
        avg_times_converged = np.sum(run_times * convergences, axis=1) / np.sum(convergences, axis=1)

        # Calculate the overall average time for trials that converged, removing nan values
        overall_avg_time_converged = np.nanmean(avg_times_converged)

        # Calcuate the overall average convergence rate
        overall_avg_convergence_rate = np.mean(convergence_rates)

        # Save the results to a csv file
        with open("results3.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Start Location", "Convergence Rate", "Average Time", "Average Time Converged"])
            for i in range(len(self.test_starts)):
                writer.writerow([i, convergence_rates[i], avg_times[i], avg_times_converged[i]])

            writer.writerow(["Overall Average Time Converged", overall_avg_time_converged])
            writer.writerow(["Overall Average Convergence Rate", overall_avg_convergence_rate])

        # Print the results to the console
        self.qt_window.console.appendPlainText("Results:")
        for i in range(len(self.test_starts)):
            self.qt_window.console.appendPlainText("Start Location: {}   Convergence Rate: {}   Average Time: {}   Average Time Converged: {}".format(
                i, convergence_rates[i], avg_times[i], avg_times_converged[i]
            ))
        self.qt_window.console.appendPlainText("Overall Average Time Converged: {}".format(overall_avg_time_converged))
        self.qt_window.console.appendPlainText("Overall Average Convergence Rate: {}".format(overall_avg_convergence_rate))

    def include_width_changed(self):
        self.pf_engine.include_width = self.qt_window.include_width_checkbox.isChecked()

    def use_saved_data_changed(self):
        self.use_loaded_data = self.qt_window.use_saved_data_checkbox.isChecked()
        if not self.use_loaded_data:
            # Get the bag file names and sort them alphabetically
            self.data_file_names = os.listdir(self.bag_file_dir)
            self.data_file_names.sort()

            # Clear the combo box
            self.qt_window.data_file_selector.clear()

            # put them in the combo box
            for file in self.data_file_names:
                self.qt_window.data_file_selector.addItem(file)

        else:
            self.data_file_names = os.listdir(self.saved_data_dir)
            self.data_file_names.sort()
            self.qt_window.data_file_selector.clear()
            for file in self.data_file_names:
                self.qt_window.data_file_selector.addItem(file)
            self.qt_window.data_file_selector.setCurrentIndex(0)
            self.open_saved_data()

    def show_image_changed(self):
        self.show_loaded_img = self.qt_window.show_image_checkbox.isChecked()

    def show_plot_changed(self):
        self.show_plot_updates = self.qt_window.update_plot_checkbox.isChecked()

    def save_data_checkbox_changed(self):
        self.save_data = self.qt_window.save_data_checkbox.isChecked()

    def mode_changed(self):

        if self.qt_window.mode_selector.currentText() == "Scroll Images":
            self.qt_window.data_file_selector.setEnabled(True)
            self.qt_window.data_file_open_button.setEnabled(True)
            self.qt_window.prev_img_button.setEnabled(True)
            self.qt_window.next_img_button.setEnabled(True)
            self.qt_window.play_fwd_button.setEnabled(True)
            self.qt_window.bag_time_line.setReadOnly(False)
            self.qt_window.continue_button.setEnabled(False)
            self.qt_window.test_selector.setEnabled(False)

            self.qt_window.start_stop_button.setText("Stop")
            self.start_stop_button_clicked()
            self.reset_app()

            self.qt_window.start_stop_button.setEnabled(False)

        else:
            self.qt_window.data_file_selector.setEnabled(False)
            self.qt_window.data_file_open_button.setEnabled(False)
            self.qt_window.prev_img_button.setEnabled(False)
            self.qt_window.next_img_button.setEnabled(False)
            self.qt_window.play_fwd_button.setEnabled(False)
            self.qt_window.bag_time_line.setReadOnly(True)

            self.qt_window.test_selector.setEnabled(False)
            self.qt_window.continue_button.setEnabled(False)


            if self.qt_window.mode_selector.currentText() == "Continuous":
                self.qt_window.start_stop_button.setEnabled(True)
            elif self.qt_window.mode_selector.currentText() == "Tests":
                self.qt_window.test_selector.setEnabled(True)
                self.qt_window.start_stop_button.setEnabled(True)
                self.test_start_num = 0
                self.load_next_trial()
            else:
                self.qt_window.start_stop_button.setEnabled(False)
                self.qt_window.continue_button.setEnabled(True)


    def load_next_trial(self):

        test_info = self.test_starts[self.test_start_num]

        data_file_name = "run_" + str(int(test_info["run_num"])) + "_gt_data.json"

        # find the index of the bag file in the list of bag files
        data_file_index = self.qt_window.data_file_selector.findText(data_file_name)

        # set the bag file selector to the correct bag file
        self.qt_window.data_file_selector.setCurrentIndex(data_file_index)

        # open the data file
        self.open_saved_data()

        # set the bag time to the correct time
        self.qt_window.bag_time_line.setText(str(test_info["start_time"]))
        self.time_stamp_line_edited()
        # set the start pose center to the correct center
        self.qt_window.start_x_input.setText(str(test_info["start_pose_center"][0]))
        self.qt_window.start_y_input.setText(str(test_info["start_pose_center"][1]))
        # set the start pose height, width, and rotation to the correct values
        self.qt_window.width_input.setText(str(test_info["start_pose_width"]))
        self.qt_window.height_input.setText(str(test_info["start_pose_height"]))
        self.qt_window.rotation_input.setText(str(test_info["start_pose_rotation"]))

        # Reset the app
        self.reset_app()

    def run_pf(self):
        while self.pf_active:
            self.send_next_msg()

            if self.cur_data_pos >= len(self.msg_order):
                continue

            converged = False
            if self.qt_window.stop_when_converged.isChecked() or self.qt_window.signal_test_trees_checkbox.isChecked():
                start_time_dist = time.time()
                if self.use_loaded_data:
                    if self.pf_engine.histogram is not None and self.img_data[self.cur_img_pos] \
                    is not None and self.msg_order[self.cur_data_pos] == 1:
                        converged = self.check_convergence(self.pf_engine.histogram)
                else:
                    if self.pf_engine.histogram is not None and self.msg_order[self.cur_data_pos] == 1:
                        converged = self.check_convergence(self.pf_engine.histogram)

            if self.qt_window.signal_test_trees_checkbox.isChecked():
                if converged and not self.converged_once:
                    self.converged_once = True
                    # best_guess = self.pf_engine.particles.mean(axis=0)
                    # self.qt_window.plotter.initialize_best_guess(best_guess)

            elif self.qt_window.stop_when_converged.isChecked() and not self.qt_window.signal_test_trees_checkbox.isChecked():
                if converged:
                    self.pf_active = False
                else:
                    self.round_exclude_time += time.time() - start_time_dist


            # if self.pf_engine.histogram is not None and (self.qt_window.stop_when_converged.isChecked() or
            #                                              self.qt_window.signal_test_trees_checkbox.isChecked()):
            #     self.converged = True
            #     # This is just a dumb stuff i had to add because i did the img_data different for the loaded data,
            #     # and i did idk, something wierd here, but it works now
            #     if self.use_loaded_data:
            #         if self.img_data[self.cur_img_pos] is None or self.msg_order[self.cur_data_pos] != 1:
            #             self.converged = False
            #     elif self.cur_data_pos >= len(self.msg_order):
            #         self.converged = False
            #     elif not self.use_loaded_data and self.msg_order[self.cur_data_pos] != 1:
            #         self.converged = False



    def update_img_label(self):
        if self.use_loaded_data:
            self.qt_window.img_number_label.setText("Image: " + str(self.cur_img_pos + 1) + "/" + str(len(self.img_data)))
        else:
            self.qt_window.img_number_label.setText("Image: " + str(self.cur_img_pos + 1) + "/" + str(len(self.paired_imgs)))

    def check_end_of_data(self):
        if self.cur_data_pos >= len(self.msg_order) and self.use_loaded_data:
            if len(self.img_data) == 0:
                # print message to console
                self.qt_window.console.appendPlainText("Please load a data file")
                # stop the PF
                self.start_stop_button_clicked()
                return
            else:
                self.qt_window.console.appendPlainText("Finished all data currently loaded.")
                self.cur_img_pos -= 1
                self.start_stop_button_clicked()
                return
        elif self.cur_data_pos >= len(self.msg_order) and not self.use_loaded_data:
            if self.cur_bag_file_name is None:
                # print message to console
                self.qt_window.console.appendPlainText("Please choose a starting bag file")
                # stop the PF
                self.start_stop_button_clicked()
                return
            # find the position of the current bag file in the list of bag files
            cur_bag_file_pos = self.data_file_names.index(self.cur_bag_file_name)
            # If the current bag file is the last one, stop the PF
            if cur_bag_file_pos == len(self.data_file_names) - 1:
                self.start_stop_button_clicked()
                self.qt_window.console.appendPlainText("Finished all bag files.")
                return
            else:
                self.open_bag_file(self.data_file_names[cur_bag_file_pos + 1])
    def send_next_msg(self):
        self.check_end_of_data()

        # Write the current time stamp to the line edit
        self.qt_window.bag_time_line.setText(str(self.time_stamps[self.cur_data_pos]))

        if self.msg_order[self.cur_data_pos] == 0:
            # while self.msg_order
            if self.use_loaded_data:
                x_odom = self.odom_data[self.cur_odom_pos]['x_odom']
                theta_odom = self.odom_data[self.cur_odom_pos]['theta_odom']
                time_stamp_odom = self.odom_data[self.cur_odom_pos]['time_stamp']
                # self.pf_engine.save_odom_loaded(x_odom, theta_odom, time_stamp_odom)
                self.pf_engine.save_odom(x_odom, theta_odom, time_stamp_odom)

            else:
                x_odom = self.odom_msgs[self.cur_odom_pos].twist.twist.linear.x
                theta_odom = self.odom_msgs[self.cur_odom_pos].twist.twist.angular.z
                time_stamp_odom = self.odom_msgs[self.cur_odom_pos].header.stamp.to_sec()

                self.pf_engine.save_odom(x_odom, theta_odom, time_stamp_odom)

                if self.save_data:
                    timestamp = self.odom_msgs[self.cur_odom_pos].header.stamp.to_sec()
                    timestamp = str(int(1000*timestamp))
                    self.saved_data[timestamp] = {'x_odom': self.odom_msgs[self.cur_odom_pos].twist.twist.linear.x,
                                                  'theta_odom': self.odom_msgs[self.cur_odom_pos].twist.twist.angular.z,
                                                  'time_stamp': self.odom_msgs[self.cur_odom_pos].header.stamp.to_sec(),
                                                  }

            self.cur_odom_pos += 1
            self.cur_data_pos += 1

        # if self.msg_order[self.cur_data_pos] == 0:
        #     if self.first_odom:
        #         self.first_odom = False
        #         self.x_odom = 0
        #         self.theta_odom = 0
        #         self.odom_counter = 0
        #     self.x_odom += self.odom_msgs[self.cur_odom_pos].twist.twist.linear.x
        #     self.theta_odom += self.odom_msgs[self.cur_odom_pos].twist.twist.angular.z
        #     self.time_stamp_odom = self.odom_msgs[self.cur_odom_pos].header.stamp.to_sec()
        #     self.odom_counter += 1
        #
        #     self.cur_data_pos += 1
        #     self.cur_odom_pos += 1


        elif self.msg_order[self.cur_data_pos] == 1:
            if self.use_loaded_data:
                img_data = self.img_data[self.cur_img_pos]
                time_stamp = self.time_stamps_keys[self.cur_data_pos]
                self.load_saved_img(time_stamp)
                self.cur_img_pos += 1
                self.cur_data_pos += 1

                if img_data is not None:
                    tree_positions = np.array(img_data['tree_data']['positions'])
                    widths = np.array(img_data['tree_data']['widths'])
                    classes = np.array(img_data['tree_data']['classes'])
                    img_x_positions = np.array(img_data['tree_data']['img_x_positions'])
                else:
                    tree_positions = None
                    widths = None
                    classes = None
                    img_x_positions = None

            else:

                # if self.first_odom:
                #     self.cur_img_pos += 1
                #     self.cur_data_pos += 1
                #     return
                #
                # if self.odom_counter > 0:
                #     x_odom = self.x_odom / self.odom_counter
                #     theta_odom = self.theta_odom / self.odom_counter
                #     print("x_odom: ", x_odom)
                #     print("theta_odom: ", theta_odom)
                #     self.pf_engine.save_odom(x_odom, theta_odom, self.time_stamp_odom)
                #
                #     self.x_odom = 0
                #     self.theta_odom = 0
                #     self.odom_counter = 0

                tree_positions, widths, classes, img_x_positions = self.get_trunk_data()

                if self.save_data:
                    timestamp = str(int(1000 * self.time_stamps_img[self.cur_img_pos]))

                self.cur_img_pos += 1
                self.cur_data_pos += 1
                self.update_img_label()

                # if tree_positions is None and self.save_data:
                #     if self.save_data:
                #         self.saved_data[timestamp] = None
                #     # return
                if tree_positions is not None:
                    # Switch sign on x_pos and y_pos
                    tree_positions[:, 0] = -tree_positions[:, 0]
                    tree_positions[:, 1] = -tree_positions[:, 1]

            if tree_positions is not None:
                tree_positions[:, 0] += self.x_offset
                tree_positions[:, 1] += self.y_offset

                # widths -= 0.01
                img_x_positions = abs(img_x_positions - 320)
                # Values obtained from calibrate_widths.py, sept = True, poly = False
                widths = -0.006246 + (-2.0884248893265422e-05 * img_x_positions) + (1.057907234666699 * widths)

                tree_data = {'positions': tree_positions, 'widths': widths, 'classes': classes}
                # self.pf_engine.save_scan(tree_data)
                self.pf_engine.scan_update(tree_data)
            else:
                self.pf_engine.resample_particles()

            # Print number of particles
            # print(self.pf_engine.particles.shape)

            if self.show_plot_updates:
                start_time = time.time()

                self.qt_window.plotter.update_particles(self.pf_engine.particles)

                # ensure plot updates by refreshing the GUI
                self.qt_app.processEvents()

                self.post_a_time(start_time, "Plotting Time: ")

            if self.save_data and not self.use_loaded_data and tree_positions is not None:
                tree_data = {'positions': tree_positions.tolist(), 'widths': widths.tolist(), 'classes': classes.tolist()}
                self.saved_data[timestamp] = {'tree_data': tree_data,
                                              'location_estimate': {}}
                best_particle = self.pf_engine.best_particle.tolist()
                self.saved_data[timestamp]['location_estimate']['x'] = best_particle[0]
                self.saved_data[timestamp]['location_estimate']['y'] = best_particle[1]
                self.saved_data[timestamp]['location_estimate']['theta'] = best_particle[2]
            elif self.save_data and not self.use_loaded_data:
                self.saved_data[timestamp] = None

            if self.converged_once:
                best_guess = self.pf_engine.particles.mean(axis=0)
                self.qt_window.plotter.update_best_guess(best_guess)
                self.find_test_trees(best_guess)

    def get_trunk_data(self):
        time_start = time.time()
        # try:
        tree_positions, widths, classes, img_seg, img_x_positions = self.trunk_analyzer.pf_helper(
            self.paired_imgs[self.cur_img_pos][1],
            self.paired_imgs[self.cur_img_pos][0],
            show_seg=True)
        # except IndexError:
        #     print("Index error")
        if self.start_time is not None and self.show_plot_updates:
            self.post_a_time(self.start_time, "Full Cycle Time: ")

        if self.show_plot_updates:
            self.post_a_time(time_start, "Time for trunk analyzer: ")

        if widths is not None and self.show_plot_updates:
            msg_str = "Widths: "
            for width in widths:
                width *= 100
                msg_str += str(round(width, 2)) + "cm,  "
            self.qt_window.console.appendPlainText(msg_str)
            msg_str = "Positions: "
            for position in tree_positions:
                msg_str += "(" + str(round(position[0], 3)) + ", " + str(round(position[1], 3)) + ") "
            self.qt_window.console.appendPlainText(msg_str)
            self.qt_window.console.appendPlainText("---")
            classes[classes == 0] = 1
            classes[classes == 2] = 0
        self.qt_window.load_image(img_seg)
        self.start_time = time.time()

        if self.save_data:
            timestamp = str(int(1000 * self.time_stamps_img[self.cur_img_pos]))
            # Save the image to file
            file_name = self.img_base_dir + "run_" + str(self.run_num) + "/" + timestamp + ".png"
            cv2.imwrite(file_name, img_seg)

        if self.save_diam_data:
            if widths is None:
                pass
            else:
                if len(widths) > 1:
                    width, img_x_position = self.multi_width_popup(widths, img_x_positions)
                else:
                    width = widths[0]
                    img_x_position = img_x_positions[0]
                # convert width to a python float and img_x_position to a python int
                width = float(width)
                img_x_position = int(img_x_position)
                self.img_x_positions.append(img_x_position)
                self.width_estimates.append(width)

        return tree_positions, widths, classes, img_x_positions

    def post_a_time(self, time_s, msg, ms=True):
        time_tot = (time.time() - time_s)
        if ms:
            time_tot = time_tot * 1000
        time_tot = round(time_tot, 1)
        if ms:
            msg_str = msg + str(time_tot) + "ms"
        else:
            msg_str = msg + str(time_tot) + "s"
        self.qt_window.console.appendPlainText(msg_str)


    def prev_button_clicked(self):

        self.cur_data_pos -= 1
        while self.msg_order[self.cur_data_pos] != 1 and self.cur_data_pos > 0:
            self.cur_data_pos -= 1
        if self.cur_data_pos <= 0:
            self.cur_data_pos = 0
            self.cur_img_pos = 0
            self.qt_window.console.appendPlainText("Reached the beginning of the bag file / loaded data")
            # return
        else:
            self.cur_img_pos -= 1
            if sum(self.msg_order[0:self.cur_data_pos]) == 0:
                self.cur_img_pos = 0

            if self.use_loaded_data:
                self.load_saved_img(self.time_stamps_keys[self.cur_data_pos])
            else:
                self.qt_window.load_image(self.paired_imgs[self.cur_img_pos][1])
                self.qt_window.bag_time_line.setText(str(self.time_stamps[self.cur_data_pos]))
        self.cur_odom_pos = max(self.cur_data_pos - sum(self.msg_order[:self.cur_data_pos]) - 1, 0)
        self.update_img_label()

    def play_button_clicked(self):
        if self.qt_window.play_fwd_button.text() == "Play":
            self.qt_window.play_fwd_button.setText("Stop")
            while self.cur_data_pos < len(self.msg_order) - 1 and self.qt_window.play_fwd_button.text() == "Stop":
                self.next_button_clicked()
                time.sleep(0.02)
            self.qt_window.play_fwd_button.setText("Play")
        else:
            self.qt_window.play_fwd_button.setText("Play")


    def next_button_clicked(self):
        self.cur_data_pos += 1
        while self.msg_order[self.cur_data_pos] != 1 and self.cur_data_pos < len(self.msg_order) - 1:
            self.cur_data_pos += 1
        if self.cur_data_pos >= len(self.msg_order) - 1:
            self.cur_data_pos = len(self.msg_order) - 1
            self.qt_window.console.appendPlainText("Reached the end of the bag file / loaded data")
            # return
        else:
            self.cur_img_pos += 1
            if sum(self.msg_order[0:self.cur_data_pos]) == 0:
                self.cur_img_pos = 0
            if self.use_loaded_data:
                self.load_saved_img(self.time_stamps_keys[self.cur_data_pos])
                img_data = self.img_data[self.cur_img_pos]
                if img_data is not None:
                    widths = np.array(img_data['tree_data']['widths'])
                    # print widths to console
                    msg_str = "Widths: "
                    for width in widths:
                        width *= 100
                        msg_str += str(round(width, 2)) + "cm,  "
                    self.qt_window.console.appendPlainText(msg_str)
            else:
                # check if this is actually the first image in the file

                self.get_segmentation()
                self.qt_window.bag_time_line.setText(str(self.time_stamps[self.cur_data_pos]))

        self.cur_odom_pos = max(self.cur_data_pos - sum(self.msg_order[:self.cur_data_pos]) - 1, 0)
        self.update_img_label()

    def get_segmentation(self):
        _, _, _, _ = self.get_trunk_data()

    def load_saved_img(self, time_stamp):
        if self.show_loaded_img:
            file_name = self.img_base_dir + "run_" + str(self.run_num) + "/" + time_stamp + ".png"

            img = cv2.imread(file_name)
            self.qt_window.load_image(img)
            self.qt_window.bag_time_line.setText(str(self.time_stamps[self.cur_data_pos]))
        self.update_img_label()

    def cont_button_clicked(self):
        # check the mode
        if self.qt_window.mode_selector.currentText() == "Manual - single step":
            self.send_next_msg()
        elif self.qt_window.mode_selector.currentText() == "Manual - single image":
            self.send_next_msg()
            while self.msg_order[self.cur_data_pos - 1] != 1:
                self.send_next_msg()

    def time_stamp_line_edited(self):
        # Get the value from the line edit
        time_stamp = self.qt_window.bag_time_line.text()
        # Check if the value is a number
        try:
            time_stamp = float(time_stamp)
        except ValueError:
            self.qt_window.console.appendPlainText("Invalid time stamp")
            return
        # Find the position of the time stamp in the list of time stamps
        time_stamp_pos = bisect.bisect_left(self.time_stamps, time_stamp)
        # Check if the position is valid
        if time_stamp_pos >= len(self.time_stamps):
            self.qt_window.console.appendPlainText("Time stamp is too large")
            return

        # Set the time line to be the time stamp
        self.qt_window.bag_time_line.setText(str(self.time_stamps[time_stamp_pos]))

        # Set the current position to the position of the time stamp
        self.cur_data_pos = time_stamp_pos

        while self.msg_order[self.cur_data_pos] != 1 and self.cur_data_pos > 0:
            self.cur_data_pos -= 1
        if self.cur_data_pos <= 0:
            self.cur_data_pos = 0
            self.cur_img_pos = 0
            self.cur_odom_pos = 0
        else:
            self.cur_img_pos = sum(self.msg_order[:self.cur_data_pos])
            self.cur_odom_pos = self.cur_data_pos - self.cur_img_pos - 1

            if self.use_loaded_data:
                self.load_saved_img(self.time_stamps_keys[self.cur_data_pos])
            else:
                self.qt_window.load_image(self.paired_imgs[self.cur_img_pos][1])

    def open_bag_file_button_clicked(self):
        if self.use_loaded_data:
            self.open_saved_data()
            return

        bag_file_name = self.qt_window.data_file_selector.currentText()
        self.open_bag_file(bag_file_name, from_button=True)

    def open_bag_file(self, bag_file_name, from_button=False):
        start_time_open = time.time()

        self.cur_bag_file_name = bag_file_name

        if not from_button:
            self.qt_window.data_file_selector.setCurrentText(self.cur_bag_file_name)

        try:
            self.run_num = int(self.cur_bag_file_name.split("_")[0].split("-")[-1])
        except ValueError:
            print("Invalid bag file name, no run number found")
            self.run_num = 0
        # Change the button text
        self.qt_window.data_file_open_button.setText("---")

        # Make sure the change is visible
        self.qt_window.data_file_open_button.repaint()

        def pair_messages(d_msg, img_msg):
            if d_msg is not None and img_msg is not None and d_msg.header.stamp == img_msg.header.stamp:
                try:
                    depth_image = self.bridge.imgmsg_to_cv2(d_msg, "passthrough")
                    color_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                    self.time_stamps_img.append(d_msg.header.stamp.to_sec())

                    # Check if color image height and width are divisible by 32
                    if color_img.shape[0] % 32 != 0:
                        color_img = color_img[:-(color_img.shape[0] % 32), :, :]
                        depth_image = depth_image[:-(depth_image.shape[0] % 32), :]
                    if color_img.shape[1] % 32 != 0:
                        color_img = color_img[:, :-(color_img.shape[1] % 32), :]
                        depth_image = depth_image[:, :-(depth_image.shape[1] % 32)]
                except CvBridgeError as e:
                    print(e)
                return (depth_image, color_img)
            else:
                return None

        path = self.bag_file_dir + self.cur_bag_file_name
        try:
            bag_data = rosbag.Bag(path)
        except IsADirectoryError:
            print("Invalid bag file name")
            self.qt_window.data_file_open_button.setText("Open")
            return
        depth_msg = None
        color_msg = None
        self.msg_order = []
        self.paired_imgs = []
        self.odom_msgs = []
        self.time_stamps = []
        self.time_stamps_img = []
        self.cur_data_pos = 0
        self.cur_odom_pos = 0
        self.cur_img_pos = 0

        t_start = None
        for topic, msg, t in bag_data.read_messages(topics=self.topics):
            if t_start is None:
                t_start = t.to_sec()
            if topic == self.topics[1]:
                depth_msg = msg
                paired_img = pair_messages(depth_msg, color_msg)
                if paired_img is not None:
                    self.paired_imgs.append(paired_img)
                    self.msg_order.append(1)
                    self.time_stamps.append(t.to_sec() - t_start)

            elif topic == self.topics[0]:
                color_msg = msg
                paired_img = pair_messages(depth_msg, color_msg)
                if paired_img is not None:
                    self.paired_imgs.append(paired_img)
                    self.msg_order.append(1)
                    self.time_stamps.append(t.to_sec() - t_start)


            elif topic == self.topics[2]:
                self.odom_msgs.append(msg)
                self.msg_order.append(0)
                self.time_stamps.append(t.to_sec() - t_start)

        self.qt_window.data_file_open_button.setText("Open")

        if from_button:
            self.qt_window.bag_time_line.setText(str(self.time_stamps[0]))
            self.time_stamp_line_edited()
            self.get_segmentation()

        # Write a message to the console
        self.qt_window.console.appendPlainText("Opened bag file: " + self.cur_bag_file_name)
        self.qt_window.console.appendPlainText("Number of Odom messages: " + str(len(self.odom_msgs)))
        self.qt_window.console.appendPlainText("Number of images: " + str(len(self.paired_imgs)))

        self.round_exclude_time += time.time() - start_time_open

    def open_saved_data(self):
        self.time_stamps = []
        self.msg_order = []
        self.cur_odom_pos = 0
        self.cur_img_pos = 0
        self.cur_data_pos = 0
        self.img_data = []
        self.odom_data = []
        t_start = None

        file_name = self.qt_window.data_file_selector.currentText()
        self.run_num = int(file_name.split("_")[1])
        data_file_path = self.saved_data_dir + file_name

        loaded_data = json.load(open(data_file_path))
        # get all the time stamps from the data, which are the keys
        self.time_stamps_keys = list(loaded_data.keys())
        self.time_stamps_keys.sort()

        # Find the last None value in loaded_data and remove it, all data after it, and all odom data between it and
        # the previous image
        last_img = 0
        for i, time_stamp_key in enumerate(self.time_stamps_keys):
            if loaded_data[time_stamp_key] is None:
                continue
            data_keys = list(loaded_data[time_stamp_key].keys())
            if 'tree_data' in data_keys:
                last_img = i

        for i, time_stamp_key in enumerate(self.time_stamps_keys):
            if i > last_img:
                break
            if t_start is None:
                t_start = float(time_stamp_key)/1000.0
            self.time_stamps.append(float(time_stamp_key)/1000.0 - t_start)
            if loaded_data[time_stamp_key] is None:
                self.msg_order.append(1)
                self.img_data.append(None)
                continue
            data_keys = list(loaded_data[time_stamp_key].keys())
            if 'x_odom' in data_keys:
                self.msg_order.append(0)
                self.odom_data.append(loaded_data[time_stamp_key])
            else:
                self.msg_order.append(1)
                self.img_data.append(loaded_data[time_stamp_key])


    def toggle_plot_nums(self):
        if self.qt_window.plotter.show_nums:
            self.qt_window.plotter.show_nums = False
            self.qt_window.plotter.draw_plot(particles=self.pf_engine.particles)
            self.qt_window.plot_nums_toggle_button.setText("Show Nums")
        else:
            self.qt_window.plotter.show_nums = True
            self.qt_window.plotter.draw_plot(particles=self.pf_engine.particles)
            self.qt_window.plot_nums_toggle_button.setText("Remove Nums")

    def converged_popup(self):
        # Ask the user if the filter converged or not
        msg_box = QMessageBox()
        msg_box.setText("Did the filter converge Correctly?")
        msg_box.setWindowTitle("Convergence")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)
        ret = msg_box.exec_()
        if ret == QMessageBox.Yes:
            return True
        else:
            return False

    def save_data_button_clicked(self):
        # Save self.saved_data to a json file
        file_name = self.saved_data_dir + 'run_' + str(self.run_num) + '_gt_data.json'
        with open(file_name, 'w') as outfile:
            json.dump(self.saved_data, outfile)

    def check_converged_location(self):
        current_position = self.pf_engine.best_particle

        # Calculate sin and cos of particle angles
        s = np.sin(current_position[2])
        c = np.cos(current_position[2])

        current_position_cam = np.zeros(2)
        current_position_cam[0] = current_position[0] + self.x_offset * c + self.y_offset * -s
        current_position_cam[1] = current_position[1] + self.x_offset * s + self.y_offset * c

        actual_position_x = self.img_data[self.cur_img_pos]['location_estimate']['x']
        actual_position_y = self.img_data[self.cur_img_pos]['location_estimate']['y']
        actual_position = np.array([actual_position_x, actual_position_y])
        distance = np.linalg.norm(current_position_cam - actual_position)
        if distance < self.convergence_threshold:
            return True, distance
        else:
            return False, distance

    def check_convergence(self, hist):
        # Convert the histogram to binary where bins with any count are considered as occupied.
        binary_mask = (hist > 0).astype(int)

        # Define a structure for direct connectivity in 3D.
        structure = np.array([[[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]],

                              [[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]],

                              [[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]])

        # Label connected components. The structure defines what is considered "connected".
        labeled_array, num_features = label(binary_mask, structure=structure)

        # If there is only one feature, then the particles have converged
        if num_features == 1:
            # Get the size of each feature
            feature_sizes = np.bincount(labeled_array.ravel())[1:]
            feature_size_max = int(((1/self.pf_engine.bin_size)**2) * ((0.25 * np.pi) / self.pf_engine.bin_angle))
            # print("Feature Sizes: ", feature_sizes)
            # print("Feature Size Max: ", feature_size_max)
            if feature_sizes[0] < feature_size_max:
                return True
            else:
                return False
        else:
            return False

    def save_diam_data_button_clicked(self):
        if self.qt_window.save_diam_data_button.text() == "Begin Saving":

            self.qt_window.save_diam_data_button.setText("Done Saving")
            self.save_diam_data = True
            self.width_estimates = []
            self.img_x_positions = []
        else:
            tree_num = int(self.qt_window.tree_num_line.text())
            diam_data = {'width_estimates': self.width_estimates, 'img_x_positions': self.img_x_positions, 'tree_num': tree_num}
            # Save the data to a file
            file_name = self.save_diam_data_dir + self.diam_data_set_name + str(tree_num) + '.json'
            with open(file_name, 'w') as outfile:
                json.dump(diam_data, outfile)

            self.qt_window.save_diam_data_button.setText("Begin Saving")
            self.qt_window.tree_num_line.setText("")
            self.save_diam_data = False

    def multi_width_popup(self, width_list, img_x_positions):
        # Ask the user which width to use given the list of widths and the corresponding image x positions
        msg_box = QMessageBox()
        msg_box.setText("Which width should be used?, Options: " + str(width_list))
        msg_box.setWindowTitle("Multiple Widths")
        buttons = []
        for i in range(len(width_list)):
            img_x_position = str(round(img_x_positions[i], 0))
            buttons.append(msg_box.addButton(img_x_position, QMessageBox.ActionRole))
        msg_box.setDefaultButton(buttons[0])
        msg_box.exec_()


        for i in range(len(width_list)):
            if msg_box.clickedButton() == buttons[i]:
                print("Width: ", width_list[i], "Img X Position: ", img_x_positions[i])
                return width_list[i], img_x_positions[i]

    def find_test_trees(self, robot_position):

        distances, idx_kd = self.test_kdtree.query(robot_position[:2], k=10)

        test_tree_positions = self.test_tree_positions[idx_kd]

        x_offsets = test_tree_positions[:,0] - robot_position[0]
        y_offsets = test_tree_positions[:,1] - robot_position[1]
        angles= np.arctan2(y_offsets, x_offsets) - robot_position[2]

        # Correct angles to be between -pi and pi
        angles[angles > np.pi] -= 2 * np.pi
        angles[angles < -np.pi] += 2 * np.pi

        # Get the position of the trees relative to the robot, x is forward, y is left
        test_tree_positions_rel_robot = np.zeros(test_tree_positions.shape)
        test_tree_positions_rel_robot[:,0] = distances * np.cos(angles)
        test_tree_positions_rel_robot[:,1] = distances * np.sin(angles)

        # Keep only trees that are to the right of the robot, and greater than -2 meters in the -y direction
        idx = np.where((angles < 0) & (test_tree_positions_rel_robot[:,1] > -2))[0]
        test_tree_positions_rel_robot = test_tree_positions_rel_robot[idx]
        idx_kd = idx_kd[idx]

        # for i in range(len(test_tree_nums)):
        #     print("Tree Num: ", test_tree_nums[i], "Position: ", np.round(test_tree_positions_rel_robot[i], 2))

        # Keep only trees with an x distance between 0.5 and 2 meters
        idx_front = np.where((test_tree_positions_rel_robot[:,0] > 0.5) & (test_tree_positions_rel_robot[:,0] < 2))[0]
        idx_kd_front = idx_kd[idx_front]

        idx_behind = np.where(test_tree_positions_rel_robot[:, 0] < -2)[0]
        idx_kd_behind = idx_kd[idx_behind]

        idx_on_side = np.where((test_tree_positions_rel_robot[:,0] > -2) & (test_tree_positions_rel_robot[:,
                                                                            0] < 2))[0]
        idx_kd_on_side = idx_kd[idx_on_side]

        if len(idx_front) > 1:
            print("Multiple test trees !?!?!?")
        elif len(idx_front) == 1:
            if self.treatment_status[idx_kd_front] == 0:
                print("Approaching tree: ", self.test_tree_nums[idx_kd_front])
                self.treatment_status[idx_kd_front] = 1
                self.qt_window.plotter.update_in_progress_tree(self.test_tree_positions[idx_kd_front][0])

        for i in idx_on_side:
            print("Tree in progress num: ", self.test_tree_nums[idx_kd_on_side[i]], "Position: ", np.round(
                test_tree_positions_rel_robot[i,0], 2), " m")

        for i in idx_kd_behind:
            if self.treatment_status[i] == 1:
                self.treatment_status[i] = 2
                print("Completed tree: ", self.test_tree_nums[i])

        complete_tree_positions = self.test_tree_positions[self.treatment_status == 2]
        self.qt_window.plotter.update_complete(complete_tree_positions)





if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = MyMainWindow()
    particle_filter = ParticleFilterBagFiles(app)
    particle_filter.qt_window.show()

    sys.exit(app.exec_())





