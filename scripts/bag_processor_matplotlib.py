#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
import os
import rosbag
import rospy
# from trunk_width_estimation.trunk_analyzer import TrunkAnalyzer
from width_estimation import TrunkAnalyzer
import json
import numpy as np
# from helper_funcs import get_map_data, ParticleMapPlotter, MyMainWindow
from pf_engine_og import PFEngineOG
import cv2
import time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QSlider, QComboBox, QSizePolicy, QPlainTextEdit, QCheckBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import sys
import bisect
import time

# To run in pycharm add the following environment variable:
# LD_PRELOAD: Set it to /usr/lib/x86_64-linux-gnu/libstdc++.so.6

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
    tree_nums = []

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
        tree_nums.append(tree['tree_num'])

    # Make the classes and positions numpy arrays
    classes = np.array(classes, dtype=int)
    positions = np.array(positions)
    widths = np.array(widths)


    # # Remove the sprinklers
    # positions = positions[classes != 2]
    # widths = widths[classes != 2]
    # classes = classes[classes != 2]

    if move_origin:
        # Find the min x and y values, subtract 5 and set that as the origin
        x_min = np.min(positions[:, 0]) - origin_offset
        y_min = np.min(positions[:, 1]) - origin_offset

        # Subtract origin from positions
        positions[:, 0] -= x_min
        positions[:, 1] -= y_min

    map_data = {'classes': classes, 'positions': positions, 'widths': widths, 'tree_nums': tree_nums}

    return map_data


class ParticleMapPlotter(QMainWindow):
    def __init__(self, map_data, figure, include_nums=True):
        super().__init__()

        self.show_nums = True

        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.classes = map_data['classes']
        self.positions = map_data['positions']
        self.tree_numbers = map_data['tree_nums']
        self.draw_plot()

    def draw_plot(self, particles=None):

        self.figure.clear()

        # Set text size
        plt.rcParams.update({'font.size': 22})
        ax = self.figure.add_subplot(111)

        row_num_xs = [4.9, 5.5, 6.05, 6.65, 7.4, 7.45, 7.9, 8.65, 9.2, 9.65, 10.25, 10.65, 11.05, 11.6, 12.1, 12.65,
                      13.2]
        row_num_ys = [4.9, 10.8, 16.5, 22.35, 28.1, 33.3, 39.05, 45.05, 51.05, 56.5, 62.6, 68.25, 73.9, 79.55, 85.6,
                      91.5, 97.3]
        row_nums = [i for i in range(len(row_num_xs))]


        if self.show_nums:
            for i, (x, y) in enumerate(self.positions):
                ax.text(x - 0.5, y-.01, str(self.tree_numbers[i]), fontsize=12)

            for i, (x, y) in enumerate(zip(row_num_xs, row_num_ys)):
                ax.text(x - 2, y-.01, str(row_nums[i]), fontsize=20)

        tree_positions = self.positions[self.classes == 0]
        post_positions = self.positions[self.classes == 1]

        ax.scatter(tree_positions[:, 0], tree_positions[:, 1], s=30, c='g', label='Trees')
        ax.scatter(post_positions[:, 0], post_positions[:, 1], s=30, c='r', label='Posts')
        ax.scatter([7], [43], s=30, c='b', label='start')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.set_title('Particle Filter Visualization')

        # Set the plot to be equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Set the plot to show gridlines
        ax.grid(True)

        self.particle_plot, = ax.plot([], [], 'b.', markersize=3, label='Particles')

        self.canvas.draw()

        if particles is not None:
            self.update_particles(particles)





    def update_particles(self, particles):

        if particles is not None:
            self.particle_plot.set_data(particles[:, 0], particles[:, 1])
        else:
            self.particle_plot.set_data([], [])
        self.canvas.draw()

class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle("My PyQt App")
        self.setGeometry(0, 0, 3500, 2500)

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

        # Setup radius input layout
        self.radius_input = QLineEdit()
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        radius_layout.addWidget(self.radius_input)

        # # Setup direction layout
        # self.direction_start_slider = QSlider(Qt.Horizontal)
        # self.direction_end_slider = QSlider(Qt.Horizontal)
        # direction_layout = QHBoxLayout()
        # direction_layout.addWidget(QLabel("Direction:"))
        # direction_layout.addWidget(self.direction_start_slider)
        # direction_layout.addWidget(self.direction_end_slider)

        # Setup num particles layout
        self.num_particles_input = QLineEdit()
        num_particles_layout = QHBoxLayout()
        num_particles_layout.addWidget(QLabel("Num Particles:"))
        num_particles_layout.addWidget(self.num_particles_input)

        # Setup width checkbox
        self.include_width_checkbox = QCheckBox("Include width in weight calculation")
        self.include_width_checkbox.setChecked(True)
        width_layout = QHBoxLayout()
        width_layout.addWidget(self.include_width_checkbox)
        # Alight the checkbox to the right
        width_layout.addStretch(1)

        # Add a selection for the mode
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Scroll Images")
        self.mode_selector.addItem("Continuous")
        self.mode_selector.addItem("Manual - single step")
        self.mode_selector.addItem("Manual - single image")

        mode_selector_layout = QHBoxLayout()
        # Add label with set width
        mode_label = QLabel("Mode:")
        mode_label.setFixedWidth(100)
        mode_selector_layout.addWidget(mode_label)
        mode_selector_layout.addWidget(self.mode_selector)

        self.continue_button = QPushButton("Continue")
        self.start_stop_button = QPushButton("Start")
        self.start_continue_layout = QHBoxLayout()
        self.start_continue_layout.addWidget(self.start_stop_button)
        self.start_continue_layout.addWidget(self.continue_button)

        self.picture_label = QLabel(self)
        self.picture_label.resize(1280, 960)
        self.picture_label.setAlignment(Qt.AlignCenter)

        self.img_number_label = QLabel(self)
        self.img_number_label.setAlignment(Qt.AlignCenter)

        # Setup img browsing layouts
        img_browsing_buttons_layout = QHBoxLayout()
        self.prev_img_button = QPushButton("Previous")
        self.next_img_button = QPushButton("Next")
        self.play_fwd_button = QPushButton("Play")
        self.segm_img_button = QPushButton("Segmentation")
        img_browsing_buttons_layout.addWidget(self.prev_img_button)
        img_browsing_buttons_layout.addWidget(self.next_img_button)
        img_browsing_buttons_layout.addWidget(self.play_fwd_button)
        img_browsing_buttons_layout.addWidget(self.segm_img_button)

        # Setup bag file selector
        self.bag_file_selector = QComboBox()
        bag_file_selector_layout = QHBoxLayout()
        bag_file_selector_label = QLabel("Current Bag File:")
        bag_file_selector_label.setFixedWidth(220)
        self.bag_file_open_button = QPushButton("Open")
        self.bag_file_open_button.setFixedWidth(120)
        bag_file_selector_layout.addWidget(bag_file_selector_label)
        bag_file_selector_layout.addWidget(self.bag_file_selector)
        bag_file_selector_layout.addWidget(self.bag_file_open_button)

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

        # Right-side plot
        self.figure = Figure(figsize=(12, 8))


        self.map_data = get_map_data()

        self.plotter = ParticleMapPlotter(self.map_data, self.figure)

        self.toolbar = NavigationToolbar2QT(self.plotter.canvas, self)

        # Set up layouts
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.reset_button)
        left_layout.addLayout(start_location_layout)
        left_layout.addLayout(radius_layout)
        # left_layout.addLayout(direction_layout)
        left_layout.addLayout(num_particles_layout)
        left_layout.addLayout(width_layout)
        left_layout.addLayout(mode_selector_layout)
        left_layout.addLayout(self.start_continue_layout)
        left_layout.addWidget(self.picture_label)
        left_layout.addWidget(self.img_number_label)
        left_layout.addLayout(img_browsing_buttons_layout)
        left_layout.addLayout(bag_file_selector_layout)
        left_layout.addLayout(bag_time_line_layout)
        left_layout.addWidget(self.console)
        left_layout.addWidget(self.clear_console_button)
        left_layout.addWidget(self.plot_nums_toggle_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.plotter.canvas)

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
        if img is None:
            img = np.ones((480, 640, 3), dtype=np.uint8) * 155

        image_cv2 = img
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        image_qt = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image_qt)
        pixmap_scaled = pixmap.scaled(self.picture_label.size(), Qt.KeepAspectRatio)
        self.picture_label.setPixmap(pixmap_scaled)

        # update to ensure the image is shown
        QApplication.processEvents()




class ParticleFilterBagFiles:
    def __init__(self, window):
        self.bridge = CvBridge()


        self.bag_file_dir = "/media/jostan/MOAD/research_data/2023_orchard_data/uncompressed/synced/pcl_mod/"

        # Get the bag file names and sort them alphabetically
        self.bag_file_names = os.listdir(self.bag_file_dir)
        self.bag_file_names.sort()

        # # Get the bag file paths, for now just keep the first 10
        # bag_file_paths = []
        # for bag_file_name in bag_file_names:
        #     if bag_file_name[13] == '5':
        #         bag_file_paths.append(bag_file_dir + bag_file_name)

        self.msg_order = []
        self.paired_imgs = []
        self.odom_msgs = []
        self.time_stamps = []
        self.cur_bag_pos = 0
        self.cur_odom_pos = 0
        self.cur_img_pos = 0
        self.cur_bag_file_name = None

        self.start_time = None

        self.pf_active = False

        self.qt_app = window

        # put them in the combo box
        for file in self.bag_file_names:
            self.qt_app.data_file_selector.addItem(file)

        self.start_pose_center = [12.7, 53.3]
        self.num_particles = 1000
        self.start_radius = 1.5

        self.qt_app.start_x_input.setText(str(self.start_pose_center[0]))
        self.qt_app.start_y_input.setText(str(self.start_pose_center[1]))
        self.qt_app.radius_input.setText(str(self.start_radius))
        self.qt_app.particle_density_input.setText(str(self.num_particles))

        self.map_data = get_map_data()

        self.pf_engine = PFEngineOG(self.map_data, start_pose_center=self.start_pose_center,
                                 start_pose_radius=self.start_radius,
                             num_particles=self.num_particles,)

        self.qt_app.plotter.update_particles(self.pf_engine.particles)

        self.trunk_analyzer = TrunkAnalyzer()

        self.rgb_topic = "/registered/rgb/image"
        self.depth_topic = "/registered/depth/image"
        self.topics = ["/registered/rgb/image", "/registered/depth/image", "/odometry/filtered"]

        self.qt_app.reset_button.clicked.connect(self.reset_app)

        self.qt_app.start_stop_button.clicked.connect(self.start_stop_button_clicked)

        self.qt_app.data_file_open_button.clicked.connect(self.open_bag_file_button_clicked)

        self.qt_app.continue_button.clicked.connect(self.cont_button_clicked)
        self.qt_app.continue_button.setEnabled(False)

        # Connect the mode selector
        self.qt_app.mode_selector.currentIndexChanged.connect(self.mode_changed)

        self.qt_app.prev_img_button.clicked.connect(self.prev_button_clicked)
        self.qt_app.next_img_button.clicked.connect(self.next_button_clicked)
        self.qt_app.play_fwd_button.clicked.connect(self.play_button_clicked)
        self.qt_app.segm_img_button.clicked.connect(self.segm_button_clicked)

        self.qt_app.plot_nums_toggle_button.clicked.connect(self.toggle_plot_nums)


        # Connect funtion for if bag time line is edited and enter is pressed
        self.qt_app.bag_time_line.returnPressed.connect(self.bag_time_line_edited)

        self.mode_changed()

        bag_file = "envy-trunks-05_1_converted_synced_pcl-mod.bag"
        self.open_bag_file(bag_file)




    def reset_app(self):
        start_x = float(self.qt_app.start_x_input.text())
        start_y = float(self.qt_app.start_y_input.text())
        self.start_pose_center = [start_x, start_y]
        self.start_radius = float(self.qt_app.radius_input.text())
        self.num_particles = int(self.qt_app.particle_density_input.text())
        self.pf_engine = PFEngineOG(self.map_data, start_pose_center=self.start_pose_center,
                                  start_pose_radius=self.start_radius,
                                  num_particles=self.num_particles, )
        self.pf_engine.include_width = self.qt_app.include_width_checkbox.isChecked()
        self.qt_app.plotter.update_particles(self.pf_engine.particles)

    def start_stop_button_clicked(self):
        # Check current text on button
        if self.qt_app.start_stop_button.text() == "Start":
            self.qt_app.start_stop_button.setText("Stop")
            self.pf_active = True

            if self.qt_app.mode_selector.currentText() == "Continuous":
                self.run_pf()
            else:
                self.qt_app.continue_button.setEnabled(True)
        else:
            self.qt_app.start_stop_button.setText("Start")
            self.pf_active = False

            self.qt_app.continue_button.setEnabled(False)

    def include_width_changed(self):
        self.pf_engine.include_width = self.qt_app.include_width_checkbox.isChecked()
    def mode_changed(self):

        if self.qt_app.mode_selector.currentText() == "Scroll Images":
            self.qt_app.data_file_selector.setEnabled(True)
            self.qt_app.data_file_open_button.setEnabled(True)
            self.qt_app.prev_img_button.setEnabled(True)
            self.qt_app.next_img_button.setEnabled(True)
            self.qt_app.segm_img_button.setEnabled(True)
            self.qt_app.play_fwd_button.setEnabled(True)
            self.qt_app.bag_time_line.setReadOnly(False)

            self.qt_app.start_stop_button.setText("Stop")
            self.start_stop_button_clicked()
            self.reset_app()

            self.qt_app.start_stop_button.setEnabled(False)

        else:
            self.qt_app.data_file_selector.setEnabled(False)
            self.qt_app.data_file_open_button.setEnabled(False)
            self.qt_app.prev_img_button.setEnabled(False)
            self.qt_app.next_img_button.setEnabled(False)
            self.qt_app.segm_img_button.setEnabled(False)
            self.qt_app.play_fwd_button.setEnabled(False)
            self.qt_app.bag_time_line.setReadOnly(True)

            self.qt_app.start_stop_button.setEnabled(True)

            if self.qt_app.mode_selector.currentText() == "Continuous":
                self.qt_app.continue_button.setEnabled(False)
            else:
                self.qt_app.continue_button.setEnabled(True)



    def run_pf(self):
        while self.pf_active:

            # time_start_run = time.time()
            self.send_next_msg()
            # time_ms = (time.time() - time_start_run) * 1000
            # time_ms = round(time_ms, 1)
            # msg_str = "Time for full step: " + str(time_ms) + "ms"
            # self.qt_app.console.appendPlainText(msg_str)


    def update_img_label(self):
        self.qt_app.img_number_label.setText("Image: " + str(self.cur_img_pos + 1) + "/" + str(len(self.paired_imgs)))
    def send_next_msg(self):
        if self.cur_bag_pos >= len(self.msg_order):
            if self.cur_bag_file_name is None:
                # print message to console
                self.qt_app.console.appendPlainText("Please choose a starting bag file")
                self.start_stop_button_clicked()
                return
            # find the position of the current bag file in the list of bag files
            cur_bag_file_pos = self.bag_file_names.index(self.cur_bag_file_name)
            # If the current bag file is the last one, stop the PF
            if cur_bag_file_pos == len(self.bag_file_names) - 1:
                self.start_stop_button_clicked()
                self.qt_app.console.appendPlainText("Finished all bag files.")
                return
            else:
                self.open_bag_file(self.bag_file_names[cur_bag_file_pos + 1])

        # Write the current time stamp to the line edit
        self.qt_app.bag_time_line.setText(str(self.time_stamps[self.cur_bag_pos]))
        if self.msg_order[self.cur_bag_pos] == 0:
            self.pf_engine.save_odom(self.odom_msgs[self.cur_odom_pos])
            # Skip plotting if there are over 300000 particles
            # if self.pf_engine.particles.shape[0] < 300000:
            #         self.qt_app.plotter.update_particles(self.pf_engine.particles)
            self.cur_odom_pos += 1
            self.cur_bag_pos += 1
        elif self.msg_order[self.cur_bag_pos] == 1:

            tree_positions, widths, classes = self.get_data()

            self.cur_img_pos += 1
            self.cur_bag_pos += 1
            self.update_img_label()

            if tree_positions is None:
                return

            # Switch sign on x_pos and y_pos
            tree_positions[:, 0] = -tree_positions[:, 0]
            tree_positions[:, 1] = -tree_positions[:, 1]

            tree_data = {'positions': tree_positions, 'widths': widths, 'classes': classes}
            self.pf_engine.save_scan(tree_data)
            self.qt_app.plotter.update_particles(self.pf_engine.particles)

    def get_data(self):
        time_start = time.time()

        tree_positions, widths, classes, img_seg = self.trunk_analyzer.pf_helper(
            self.paired_imgs[self.cur_img_pos][1],
            self.paired_imgs[self.cur_img_pos][0],
            show_seg=True)
        if self.start_time is not None:
            time_ms = (time.time() - self.start_time) * 1000
            time_ms = round(time_ms, 1)
            msg_str = "Full Cycle Time: " + str(time_ms) + "ms"
            self.qt_app.console.appendPlainText(msg_str)
        time_ms = (time.time() - time_start) * 1000
        time_ms = round(time_ms, 1)
        msg_str = "Time for trunk analyzer: " + str(time_ms) + "ms"
        self.qt_app.console.appendPlainText(msg_str)
        if widths is not None:
            msg_str = "Widths: "
            for width in widths:
                width *= 100
                msg_str += str(round(width, 2)) + "cm,  "
            self.qt_app.console.appendPlainText(msg_str)
            msg_str = "Positions: "
            for position in tree_positions:
                msg_str += "(" + str(round(position[0], 3)) + ", " + str(round(position[1], 3)) + ") "
            self.qt_app.console.appendPlainText(msg_str)
            self.qt_app.console.appendPlainText("---")
            classes[classes == 0] = 1
            classes[classes == 2] = 0
        self.qt_app.load_image(img_seg)
        self.start_time = time.time()
        return tree_positions, widths, classes

    def prev_button_clicked(self):
        self.cur_bag_pos -= 1
        while self.msg_order[self.cur_bag_pos] != 1 and self.cur_bag_pos > 0:
            self.cur_bag_pos -= 1
        if self.cur_bag_pos <= 0:
            self.cur_bag_pos = 0
            self.qt_app.console.appendPlainText("Reached the beginning of the bag file")
            # return
        else:
            self.cur_img_pos -= 1
            self.qt_app.load_image(self.paired_imgs[self.cur_img_pos][1])
            self.qt_app.bag_time_line.setText(str(self.time_stamps[self.cur_bag_pos]))

        self.update_img_label()

    def play_button_clicked(self):
        if self.qt_app.play_fwd_button.text() == "Play":
            self.qt_app.play_fwd_button.setText("Stop")
            while self.cur_bag_pos < len(self.msg_order) - 1 and self.qt_app.play_fwd_button.text() == "Stop":
                self.next_button_clicked()
                time.sleep(0.08)
            self.qt_app.play_fwd_button.setText("Play")
        else:
            self.qt_app.play_fwd_button.setText("Play")


    def next_button_clicked(self):
        self.cur_bag_pos += 1
        while self.msg_order[self.cur_bag_pos] != 1 and self.cur_bag_pos < len(self.msg_order) - 1:
            self.cur_bag_pos += 1
        if self.cur_bag_pos >= len(self.msg_order) - 1:
            self.cur_bag_pos = len(self.msg_order) - 1
            self.qt_app.console.appendPlainText("Reached the end of the bag file")
            # return
        else:
            self.cur_img_pos += 1

            # self.qt_app.load_image(self.paired_imgs[self.cur_img_pos][1])
            self.segm_button_clicked()
            self.qt_app.bag_time_line.setText(str(self.time_stamps[self.cur_bag_pos]))

        self.update_img_label()

    def segm_button_clicked(self):
        _, _, _, = self.get_data()

    def cont_button_clicked(self):
        # check the mode
        if self.qt_app.mode_selector.currentText() == "Manual - single step":
            self.send_next_msg()
        elif self.qt_app.mode_selector.currentText() == "Manual - single image":
            self.send_next_msg()
            while self.msg_order[self.cur_bag_pos - 1] != 1:
                self.send_next_msg()

    def bag_time_line_edited(self):
        # Get the value from the line edit
        time_stamp = self.qt_app.bag_time_line.text()
        # Check if the value is a number
        try:
            time_stamp = float(time_stamp)
        except ValueError:
            self.qt_app.console.appendPlainText("Invalid time stamp")
            return
        # Find the position of the time stamp in the list of time stamps
        time_stamp_pos = bisect.bisect_left(self.time_stamps, time_stamp)
        # Check if the position is valid
        if time_stamp_pos >= len(self.time_stamps):
            self.qt_app.console.appendPlainText("Time stamp is too large")
            return
        # Set the current position to the position of the time stamp
        self.cur_bag_pos = time_stamp_pos
        # Update the image
        self.cur_img_pos = sum(self.msg_order[:self.cur_bag_pos])
        self.qt_app.load_image(self.paired_imgs[self.cur_img_pos][1])
        self.cur_odom_pos = self.cur_bag_pos - self.cur_img_pos

    def open_bag_file_button_clicked(self):
        bag_file_name = self.qt_app.data_file_selector.currentText()
        self.open_bag_file(bag_file_name, from_button=True)

    def open_bag_file(self, bag_file_name, from_button=False):

        self.cur_bag_file_name = bag_file_name

        if not from_button:
            self.qt_app.data_file_selector.setCurrentText(self.cur_bag_file_name)

        # Change the button text
        self.qt_app.data_file_open_button.setText("---")

        # Make sure the change is visible
        self.qt_app.data_file_open_button.repaint()

        def pair_messages(d_msg, img_msg):
            if d_msg is not None and img_msg is not None and d_msg.header.stamp == img_msg.header.stamp:
                try:
                    depth_image = self.bridge.imgmsg_to_cv2(d_msg, "passthrough")
                    color_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                except CvBridgeError as e:
                    print(e)
                return (depth_image, color_img)
            else:
                return None

        path = self.bag_file_dir + self.cur_bag_file_name
        bag_data = rosbag.Bag(path)
        depth_msg = None
        color_msg = None
        self.msg_order = []
        self.paired_imgs = []
        self.odom_msgs = []
        self.time_stamps = []
        self.cur_bag_pos = 0
        self.cur_odom_pos = 0
        self.cur_img_pos = 0

        t_start = None
        for topic, msg, t in bag_data.read_messages(topics=self.topics):
            if t_start is None:
                t_start = t.to_sec()
            if topic == "/registered/depth/image":
                depth_msg = msg
                paired_img = pair_messages(depth_msg, color_msg)
                if paired_img is not None:
                    self.paired_imgs.append(paired_img)
                    self.msg_order.append(1)
                    self.time_stamps.append(t.to_sec() - t_start)

            elif topic == "/registered/rgb/image":
                color_msg = msg
                paired_img = pair_messages(depth_msg, color_msg)
                if paired_img is not None:
                    self.paired_imgs.append(paired_img)
                    self.msg_order.append(1)
                    self.time_stamps.append(t.to_sec() - t_start)

            elif topic == "/odometry/filtered":
                self.odom_msgs.append(msg)
                self.msg_order.append(0)
                self.time_stamps.append(t.to_sec() - t_start)

        self.qt_app.data_file_open_button.setText("Open")

        # Write a message to the console
        self.qt_app.console.appendPlainText("Opened bag file: " + self.cur_bag_file_name)
        self.qt_app.console.appendPlainText("Number of Odom messages: " + str(len(self.odom_msgs)))
        self.qt_app.console.appendPlainText("Number of images: " + str(len(self.paired_imgs)))

    def toggle_plot_nums(self):
        if self.qt_app.plotter.show_nums:
            self.qt_app.plotter.show_nums = False
            self.qt_app.plotter.draw_plot(particles=self.pf_engine.particles)
            self.qt_app.plot_nums_toggle_button.setText("Show Nums")
        else:
            self.qt_app.plotter.show_nums = True
            self.qt_app.plotter.draw_plot(particles=self.pf_engine.particles)
            self.qt_app.plot_nums_toggle_button.setText("Remove Nums")








#
# # Loop through the bag files
# time_prev = None
# for bag_file_path in bag_file_paths:
#
#     bag_data = rosbag.Bag(bag_file_path)
#
#     topics = ["/registered/rgb/image", "/registered/depth/image", "/odometry/filtered"]
#     # Loop through the bag file messages of topic in topics
#     depth_image = None
#     depth_msg = None
#     color_image = None
#     color_msg = None
#     skip_count = 0
#     save_directory = "/media/jostan/MOAD/research_data/2023_orchard_data/segs/"
#     save_directory += bag_file_path.split('/')[-1].split('.')[0] + '/'
#
#
#     for topic, msg, t in bag_data.read_messages(topics=topics):
#
#         if time_prev is None:
#             start_time = t
#             # Convert rospy.Time to seconds
#             start_time = start_time.to_sec()
#             time_prev = t
#         if t.to_sec() - start_time < 40:
#             continue
#
#         if topic == '/throttled/camera/depth/image_rect_raw' or topic == "/registered/depth/image":
#             try:
#                 depth_msg = msg
#                 depth_image = bridge.imgmsg_to_cv2(msg, "passthrough")
#
#             except CvBridgeError as e:
#                 print(e)
#             check_send_seg(color_image, depth_image, pf_engine, depth_msg, color_msg, skip_count)
#
#         elif topic == '/throttled/camera/color/image_raw' or topic == "/registered/rgb/image":
#             try:
#                 color_msg = msg
#                 color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
#             except CvBridgeError as e:
#                 print(e)
#
#             check_send_seg(color_image, depth_image, pf_engine, depth_msg, color_msg, skip_count)
#
#         elif topic == "/odometry/filtered":
#             pf_engine.save_odom(msg)
#             # Skip plotting if there are over 300000 particles
#             if pf_engine.particles.shape[0] < 300000:
#                 plotter.update_particles(pf_engine.particles)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    particle_filter = ParticleFilterBagFiles(window)

    window.show()

    sys.exit(app.exec_())





