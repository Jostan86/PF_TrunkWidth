#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import json
import numpy as np
from live_pf_engine_cpy import PFEngine
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QDialog, QPlainTextEdit, QCheckBox, QDialogButtonBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import sys
import time
from scipy.spatial import KDTree
from pf_trunk_width.msg import TreeInfoMulti, TestTreeLocation, TestTreeLocationMulti
import multiprocessing as mp
from live_pf_widgets import SettingsDialog, ParticleMapPlotter
from env_vars import *
import os

# To debug in pycharm add the following environment variable:
# LD_PRELOAD: Set it to /usr/lib/x86_64-linux-gnu/libstdc++.so.6

def get_map_data(include_sprinklers=False, move_origin=True, origin_offset=5):
    # Path to the tree data dictionary
    tree_data_path = os.environ.get('MAP_DATA_PATH')
    # tree_data_path = ('/home/jostan/catkin_ws/src/pkgs_noetic/research_pkgs/orchard_data_analysis/data'
    #                   '/2020_11_bag_data/afternoon2/tree_list_mod4.json')

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

        # Save test trees as a different class
        if tree['test_tree']:
            test_tree_nums.append(int(tree['test_tree_num']))
        else:
            test_tree_nums.append(None)

        # Save the data to the lists
        classes.append(tree['class_estimate'])
        # Move the tree positions by the gps adjustment
        position_estimate = np.array(tree['position_estimate']) + np.array(tree['gps_adjustment'])
        positions.append(position_estimate.tolist())
        widths.append(tree['width_estimate'])
        tree_nums.append(tree['tree_num'])

    # Convert to numpy arrays
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

    # Pack the data into a dictionary
    map_data = {'classes': classes, 'positions': positions, 'widths': widths, 'tree_nums': tree_nums, 'test_tree_nums': test_tree_nums}

    return map_data

class MyMainWindow(QMainWindow):
    def __init__(self):
        """Constructs the GUI for the particle filter app"""
        super(MyMainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle("My PyQt App")

        # This is to move the window to a specific screen and location. Commented out because I'm not sure if it works
        # on all systems
        desktop = QApplication.desktop()
        target_screen_number = 0
        if target_screen_number < desktop.screenCount():
            target_screen = desktop.screen(target_screen_number)
            self.move(target_screen.geometry().left(), target_screen.geometry().top())
        self.setGeometry(0, 0, 1900, 1050)
        # self.setGeometry(0, 0, 800, 500)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)


        # Left-side widgets

        # Button to reset the particle filter
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Reset the particle filter")
        # Button to connect to ros
        self.connect_ros_button = QPushButton("Connect to ROS")
        self.connect_ros_button.setToolTip("Connect to ROS subscribers")
        # Button to trigger popup to adjust pf settings
        self.adjust_pf_settings_button = QPushButton("PF Parameters")
        self.adjust_pf_settings_button.setToolTip("Adjust the particle filter parameters")
        top_buttons_layout = QHBoxLayout()
        top_buttons_layout.addWidget(self.reset_button)
        top_buttons_layout.addWidget(self.connect_ros_button)
        top_buttons_layout.addWidget(self.adjust_pf_settings_button)

        # Setup start location layout
        self.start_x_input = QLineEdit()
        self.start_x_input.setToolTip("X coordinate of the center of the starting block, shift click on the plot to set this")
        self.start_y_input = QLineEdit()
        self.start_y_input.setToolTip("Y coordinate of the center of the starting block, shift click on the plot to set this")
        start_location_layout = QHBoxLayout()
        start_location_layout.addWidget(QLabel("Start Pose Center:"))
        start_location_layout.addWidget(QLabel("X:"))
        start_location_layout.addWidget(self.start_x_input)
        start_location_layout.addWidget(QLabel("Y:"))
        start_location_layout.addWidget(self.start_y_input)
        self.rotation_input = QLineEdit()
        self.rotation_input.setToolTip("Rotation of the starting block in degrees, -32 aligns with the rows nicely")
        start_location_layout.addWidget(QLabel("Rotation (deg):"))
        start_location_layout.addWidget(self.rotation_input)

        # Setup start area width, height and particle density inputs
        self.width_input = QLineEdit()
        self.width_input.setToolTip("Width of the starting block")
        self.height_input = QLineEdit()
        self.height_input.setToolTip("Height of the starting block")
        self.particle_density_input = QLineEdit()
        self.particle_density_label = QLabel("Particle Density (p/m^2):")
        width_height_layout = QHBoxLayout()
        width_height_layout.addWidget(QLabel("Width (m):"))
        width_height_layout.addWidget(self.width_input)
        width_height_layout.addWidget(QLabel("Height (m):"))
        width_height_layout.addWidget(self.height_input)
        width_height_layout.addWidget(self.particle_density_label)
        width_height_layout.addWidget(self.particle_density_input)

        # Setup checkboxes
        self.include_width_checkbox = QCheckBox("Include width in weight calculation")
        self.include_width_checkbox.setChecked(True)
        self.signal_test_trees_checkbox = QCheckBox("Signal Test Trees")
        self.signal_test_trees_checkbox.setChecked(True)
        self.signal_test_trees_checkbox.setToolTip("Doesn't actually do anything yet")
        self.disable_plotting_checkbox = QCheckBox("Disable plotting until converged")
        self.disable_plotting_checkbox.setToolTip("Reduces CPU load")
        self.disable_plotting_checkbox.setChecked(True)

        checkbox_layout1 = QHBoxLayout()
        checkbox_layout1.addWidget(self.include_width_checkbox)
        checkbox_layout1.addWidget(self.signal_test_trees_checkbox)
        checkbox_layout1.addWidget(self.disable_plotting_checkbox)
        checkbox_layout1.addStretch(1)

        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.setToolTip("Start or stop the particle filter")
        self.start_layout = QHBoxLayout()
        self.start_layout.addWidget(self.start_stop_button)

        num_particles_layout = QHBoxLayout()
        num_particles_label = QLabel("Current number of particles:")
        num_particles_label.setToolTip("Number of particles currently in the particle filter")
        num_particles_layout.addWidget(num_particles_label)
        self.num_particles_label = QLabel("0")
        num_particles_layout.addWidget(self.num_particles_label)
        num_particles_layout.addStretch(1)

        queue_size_layout = QHBoxLayout()
        queue_size_label = QLabel("Current queue size:")
        queue_size_label.setToolTip("Number of messages in the queue waiting to be processed")
        queue_size_layout.addWidget(queue_size_label)
        self.queue_size_label = QLabel("0")
        queue_size_layout.addWidget(self.queue_size_label)
        queue_size_layout.addStretch(1)

        # Add an area to display the image
        self.picture_label = QLabel(self)
        self.picture_label.resize(640, 480)
        self.picture_label.setAlignment(Qt.AlignCenter)

        # Add an area to write messages
        self.console = QPlainTextEdit(self)
        self.console.setReadOnly(True)

        # Add a button to clear the console
        self.clear_console_button = QPushButton("Clear Console")
        # Add a button to remove the nums from the plot
        self.plot_nums_toggle_button = QPushButton("Show Tree and Row Numbers")
        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.addWidget(self.clear_console_button)
        bottom_buttons_layout.addWidget(self.plot_nums_toggle_button)

        # Load the map data
        self.map_data = get_map_data()

        # Create the plotter
        self.plotter = ParticleMapPlotter(self.map_data)

        # Set up layouts
        left_layout = QVBoxLayout()
        left_layout.addLayout(top_buttons_layout)
        left_layout.addLayout(start_location_layout)
        left_layout.addLayout(width_height_layout)
        left_layout.addLayout(checkbox_layout1)
        left_layout.addLayout(self.start_layout)
        left_layout.addLayout(num_particles_layout)
        left_layout.addLayout(queue_size_layout)
        left_layout.addWidget(self.picture_label)
        left_layout.addWidget(self.console)
        left_layout.addLayout(bottom_buttons_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plotter)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        central_widget.setLayout(main_layout)

        # Connect the clear console button
        self.clear_console_button.clicked.connect(self.clear_console)

        # Load a blank image into the image viewer
        self.load_image()

    def clear_console(self):
        self.console.clear()

    def load_image(self, image_cv2=None):
        """
        Load an image into the GUI image viewer
        Parameters
        ----------
        image_cv2 : OpenCV image

        Returns None
        -------

        """
        # If image is none make a blank image
        if image_cv2 is None:
            image_cv2 = np.ones((480, 640, 3), dtype=np.uint8) * 155

        # Convert the image to a Qt image and display it
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

        rospy.init_node('pf_node', anonymous=True)

        # Publisher for publishing proximity to the test trees
        self.test_tree_pub = rospy.Publisher("/orchard_pf/test_tree", TestTreeLocationMulti, queue_size=10)

        self.qt_window = MyMainWindow()
        self.qt_app = app

        # Function to initialize some startup parameters, this is just to make it easier to switch between different
        # setups
        self.initialize_startup_params1()

        # Particle filter parameters I have found to work well
        self.pf_settings = {
            'r_dist': 0.7,
            'r_angle': 25,
            # 'width_sd': 0.03,
            'width_sd': 0.04,
            'dist_sd': 0.45,
            # 'epsilon': 0.035,
            'epsilon': 0.025,
            'delta': 0.05,
            'bin_size': 0.8,
            'bin_angle': 6
        }

        # Set the time error of the odometry messages, their time stamps are not accurate, the warthog time appears to
        # be about 1.7 seconds ahead of the laptop time
        # self.odom_time_diff = 1.7
        self.odom_time_diff = 0.46

        # Initialize some variables
        self.start_time = None
        self.pf_active = False
        self.ros_connected = False

        self.plot_x_click = None
        self.plot_y_click = None

        # Set the text of the line edit input according to the startup parameters
        self.qt_window.start_x_input.setText(str(self.start_pose_center[0]))
        self.qt_window.start_y_input.setText(str(self.start_pose_center[1]))
        self.qt_window.width_input.setText(str(self.start_width))
        self.qt_window.height_input.setText(str(self.start_height))
        self.qt_window.rotation_input.setText(str(self.start_rotation))
        self.qt_window.particle_density_input.setText(str(self.particle_density))

        # Load the map data and extract the test tree data
        self.map_data = get_map_data()
        self.test_tree_nums = np.array([0 if num is None else num for num in self.map_data['test_tree_nums']])
        self.test_tree_positions = self.map_data['positions'][self.test_tree_nums > 0]
        self.test_tree_nums = self.test_tree_nums[self.test_tree_nums > 0]
        # KDTree for finding the closest test tree
        self.test_kdtree = KDTree(self.test_tree_positions)

        # Connect the buttons to their functions
        self.qt_window.reset_button.clicked.connect(self.reset_app)
        self.qt_window.start_stop_button.clicked.connect(self.start_stop_button_clicked)
        self.qt_window.plot_nums_toggle_button.clicked.connect(self.toggle_plot_nums)
        self.qt_window.connect_ros_button.clicked.connect(self.connect_ros)
        self.qt_window.plotter.plot_widget.clicked.connect(self.plot_clicked)
        self.qt_window.adjust_pf_settings_button.clicked.connect(self.adjust_pf_settings)

        # Set up the queues and processes for the particle filter
        self.pf_update_queue = mp.Queue()
        self.pf_return_queue = mp.Queue()
        # The particle filter will be run in a separate process
        self.pf_process = mp.Process(target=run_pf_process, args=(self.pf_update_queue, self.pf_return_queue, self.map_data))
        self.pf_process.start()

        self.reset_app()


    def initialize_startup_params1(self):

        # Topics to subscribe to for september data from Warthog
        self.topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", "/odometry/filtered"]

        # # Start for 72 sec into row106_107_sept.bag, at tree 795
        # self.start_pose_center = [28, 95]
        # self.particle_density = 400
        # self.start_width = 20
        # self.start_height = 20
        # self.start_rotation = 0

        # Start for 72 sec in row106_107_sept.bag, at tree 795
        # self.start_pose_center = [30.2, 97.5]
        # self.particle_density = 400
        # self.start_width = 6
        # self.start_height = 10
        # self.start_rotation = -32
        self.start_pose_center = [12.1, 53]
        self.particle_density = 400
        self.start_width = 4
        self.start_height = 20
        self.start_rotation = -32

        # Start for beginning of row106_107_sept.bag
        # self.start_pose_center = [11, 66.3]
        # self.particle_density = 500
        # self.start_width = 2
        # self.start_height = 10
        # self.start_rotation = -32


    def initialize_startup_params2(self):

        # Topics to subscribe to for data from husky
        self.topics = ["/registered/rgb/image", "/registered/depth/image", "/odometry/filtered"]

        self.start_pose_center = [16, 99.2]
        self.particle_density = 500
        self.start_width = 6
        self.start_height = 22
        self.start_rotation = -32

        # Whole map for when you're feeling arrogant
        # self.start_pose_center = [20, 70]
        # self.particle_density = 500
        # self.start_width = 60
        # self.start_height = 100
        # self.start_rotation = -32

    def reset_app(self):
        # Reset the particle filter

        # Stop the particle filter if it is running
        self.pf_active = False

        # Disconnect from ROS if connected, save the state and reconnect after if it was connected before
        if self.ros_connected:
            self.disconnect_ros()
            reconnect = True
        else:
            reconnect = False

        # Get start information from the GUI
        start_x = float(self.qt_window.start_x_input.text())
        start_y = float(self.qt_window.start_y_input.text())
        self.start_pose_center = [start_x, start_y]
        self.start_height = float(self.qt_window.height_input.text())
        self.start_width = float(self.qt_window.width_input.text())
        self.start_rotation = float(self.qt_window.rotation_input.text())
        self.particle_density = int(self.qt_window.particle_density_input.text())
        self.num_particles = int(self.particle_density * self.start_height * self.start_width)
        self.qt_window.num_particles_label.setText(str(self.num_particles))

        # Set up the data to send to the particle filter
        self.pf_setup_data = {
            'start_pose_center': self.start_pose_center,
            'start_pose_width': self.start_width,
            'start_pose_height': self.start_height,
            'rotation': self.start_rotation,
            'num_particles': self.num_particles,
            'r_dist': self.pf_settings['r_dist'],
            'r_angle': np.deg2rad(self.pf_settings['r_angle']),
            'dist_sd': self.pf_settings['dist_sd'],
            'width_sd': self.pf_settings['width_sd'],
            'epsilon': self.pf_settings['epsilon'],
            'delta': self.pf_settings['delta'],
            'bin_size': self.pf_settings['bin_size'],
            'bin_angle': np.deg2rad(self.pf_settings['bin_angle']),
            'include_width': self.qt_window.include_width_checkbox.isChecked(),
        }


        self.odom_msgs = []
        self.odom_times = []

        # Flag to track if the particle filter has converged at least once
        self.converged_once = False

        # Array to track of test trees have been treated
        self.treatment_status = np.zeros(len(self.test_tree_nums))

        # Reset the particles on the plot
        self.qt_window.plotter.update_best_guess(None)
        self.qt_window.plotter.update_in_progress_tree(None)
        self.qt_window.plotter.update_complete(None)

        # Send the setup data to the particle filter process
        self.pf_update_queue.put((None, None, self.pf_setup_data))
        try:
            # Wait for the particle filter to setup and return the particles. In a try except block to avoid getting
            # stuck, but nothing is really implemented to handle this
            self.particles, self.best_guess, _, _, _ = self.pf_return_queue.get(timeout=2)
        except mp.queues.Empty:
            print("Particle filter setup timed out")
            return

        # Update the plot
        self.qt_window.plotter.update_particles(self.particles)

        # Reconnect to ROS if it was connected before
        if reconnect:
            self.connect_ros()

        self.qt_window.console.appendPlainText("Particle filter reset")

    def connect_ros(self):
        # Connect to ROS subscribers

        # # Kinda just here to remind me i should utilize/implement TF stuff
        # self.tf_listener = tf.TransformListener()
        # self.tf_broadcaster = tf.TransformBroadcaster()
        self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        self.image_sub = rospy.Subscriber("/orchard_pf/seg_image", Image, self.image_callback)
        self.trunk_data_sub = rospy.Subscriber("/orchard_pf/trunk_data", TreeInfoMulti, self.trunk_data_callback)

        # Callback timer to get stuff out of the queue from the particle filter and plot it
        rospy.Timer(rospy.Duration(0.1), self.update_pf_status)

        self.ros_connected = True
        self.prev_tree_data_time = None

        self.qt_window.console.appendPlainText("Connected to ROS")

    def disconnect_ros(self):
        # Disconnect from ROS subscribers

        self.odom_sub.unregister()
        self.image_sub.unregister()
        self.trunk_data_sub.unregister()

        # Empty the queues
        while not self.pf_update_queue.empty():
            self.pf_update_queue.get(block=False)
        while not self.pf_return_queue.empty():
            self.pf_return_queue.get(block=False)

        self.ros_connected = False

    def odom_callback(self, odom_msg):
        # Save the odom message and time is was taken, subtracting the clock offset
        self.odom_msgs.append(odom_msg)
        self.odom_times.append(odom_msg.header.stamp.to_sec() - self.odom_time_diff)

    def trunk_data_callback(self, trunk_data_msg):
        # Save the tree data

        # Extract the tree data from the message
        widths = []
        positions = []
        classes = []

        # Class is set to 5 if no trees are detected, kind of a hack way of doing it, but it works
        if trunk_data_msg.trees[0].classification == 5:
            tree_data = {'widths': None, 'positions': None, 'classes': None, 'time_stamp': None}
        else:
            for tree_info in trunk_data_msg.trees:
                widths.append(tree_info.width)
                position = [tree_info.position.x, tree_info.position.y]
                positions.append(position)
                classes.append(tree_info.classification)
            tree_data = {'widths': widths, 'positions': positions, 'classes': classes, 'time_stamp':
                trunk_data_msg.header.stamp}

        # Set the time if this is the first tree data message
        if self.prev_tree_data_time is None:
            self.prev_tree_data_time = trunk_data_msg.header.stamp.to_sec()
            return
        else:
            # Update the particle filter with the tree data, and the time the image was taken
            self.update_pf(tree_data, trunk_data_msg.header.stamp.to_sec())

    def image_callback(self, image_msg):
        # print image to gui
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.qt_window.load_image(image)


    def update_pf(self, tree_data, tree_data_time):
        # Sort the odom messages and tree messages according to time

        # If ROS is not connected, don't do anything
        if not self.ros_connected:
            return

        # Get the odom messages that are between the previous tree data time and the current tree data time
        start_index = next((i for i, v in enumerate(self.odom_times) if v >= self.prev_tree_data_time), 0)
        end_index = next((i for i, v in enumerate(self.odom_times) if v >= tree_data_time), len(self.odom_times))
        odom_msgs_cur = self.odom_msgs[start_index:end_index]
        odom_times_cur = self.odom_times[start_index:end_index]

        # Get rid of the odom messages that have been used
        self.odom_msgs = self.odom_msgs[end_index:]
        self.odom_times = self.odom_times[end_index:]

        # Save the current tree data time for the next iteration
        self.prev_tree_data_time = tree_data_time

        # Get the average linear and angular velocities, the number of odom readings, and the time of the last odom
        # reading from the odom messages
        if len(odom_msgs_cur) > 0:
            x_odom = np.mean(np.array([odom_msg.twist.twist.linear.x for odom_msg in odom_msgs_cur]))
            theta_odom = np.mean(np.array([odom_msg.twist.twist.angular.z for odom_msg in odom_msgs_cur]))
            time_odom = np.max(np.array([odom_times_cur]))
            num_readings = len(odom_msgs_cur)

            odom_data = {'x_odom': x_odom, 'theta_odom': theta_odom, 'time_odom': time_odom, 'num_readings': num_readings}
        else:
            odom_data = None

        # If the particle filter is active, send the tree data and odom data to the particle filter
        if self.pf_active:
            self.pf_update_queue.put((odom_data, tree_data, None))

    def update_pf_status(self, event):
        # Update some GUI elements and the plot with the particle filter status

        # Get the number of items in the queue waiting to be processed by the particle filter and update the GUI
        num_items_in_queue = self.pf_update_queue.qsize()
        self.qt_window.queue_size_label.setText(str(num_items_in_queue))

        # If there is nothing in the queue, don't do anything
        if self.pf_return_queue.empty():
            return

        # Get the most recent queue item and get rid of the rest
        while not self.pf_return_queue.empty():
            self.particles, self.best_guess, self.converged, self.num_particles_cur, time_stamp = self.pf_return_queue.get(
                block=False)

        # Update the plot with the particles if the disable plotting checkbox is not checked, or if the particle filter
        # has converged at least once
        if self.qt_window.disable_plotting_checkbox.isChecked() and self.converged_once:
            self.qt_window.plotter.update_particles(self.particles)
        elif not self.qt_window.disable_plotting_checkbox.isChecked():
            self.qt_window.plotter.update_particles(self.particles)

        # Update the number of particles on the GUI
        self.qt_window.num_particles_label.setText(str(self.num_particles_cur))

        # Update the best guess on the plot if the particle filter has converged at least once
        if self.converged_once:
            self.qt_window.plotter.update_best_guess(self.best_guess)
            self.find_test_trees(self.best_guess, time_stamp)
            self.check_end_of_row(self.best_guess, time_stamp)
        else:
            # Update whether the particle filter has converged once
            if self.converged and not self.converged_once:
                self.converged_once = True

    def start_stop_button_clicked(self):
        # Method for starting and stopping the particle filter when the start/stop button is clicked

        if self.qt_window.start_stop_button.text() == "Start":
            self.qt_window.start_stop_button.setText("Stop")
            self.pf_active = True
        else:
            self.qt_window.start_stop_button.setText("Start")
            self.pf_active = False


    def post_a_time(self, time_s, msg, ms=True):
        # Method for posting a time to the console

        time_tot = (time.time() - time_s)
        if ms:
            time_tot = time_tot * 1000
        time_tot = round(time_tot, 1)
        if ms:
            msg_str = msg + str(time_tot) + "ms"
        else:
            msg_str = msg + str(time_tot) + "s"
        self.qt_window.console.appendPlainText(msg_str)


    def toggle_plot_nums(self):
        # Method for toggling the tree numbers on the plot on and off

        if self.qt_window.plotter.show_nums:
            self.qt_window.plotter.show_nums = False
            self.qt_window.plotter.draw_plot(self.particles)
            self.qt_window.plot_nums_toggle_button.setText("Show Nums")
        else:
            self.qt_window.plotter.show_nums = True
            self.qt_window.plotter.draw_plot(self.particles)
            self.qt_window.plot_nums_toggle_button.setText("Remove Nums")


    def find_test_trees(self, robot_position, time_stamp):
        # Method for finding the test trees and updating their status

        # Get the 10 closest trees to the robot
        distances, idx_kd = self.test_kdtree.query(robot_position[:2], k=10)
        test_tree_positions = self.test_tree_positions[idx_kd]

        # Get the angle to the trees
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

        # Keep only trees that are between 0 and 2 meters to the right of the robot
        idx = np.where((angles < 0) & (test_tree_positions_rel_robot[:,1] > -2))[0]
        test_tree_positions_rel_robot = test_tree_positions_rel_robot[idx]
        idx_kd = idx_kd[idx]

        # Find trees with an x distance between 0.5 and 2 meters
        idx_front = np.where((test_tree_positions_rel_robot[:,0] > 0.5) & (test_tree_positions_rel_robot[:,0] < 2))[0]
        idx_kd_front = idx_kd[idx_front]

        # Find trees beyond 2 meters behind the robot
        idx_behind = np.where(test_tree_positions_rel_robot[:, 0] < -2)[0]
        idx_kd_behind = idx_kd[idx_behind]

        # Find trees on the side of the robot
        idx_on_side = np.where((test_tree_positions_rel_robot[:,0] > -2) & (test_tree_positions_rel_robot[:, 0] < 2))[0]
        idx_kd_on_side = idx_kd[idx_on_side]

        if len(idx_front) > 1:
            # There should only be at most one test tree in this vicinity
            print("Multiple test trees !?!?!?")
        elif len(idx_front) == 1:
            # If there is a test tree between 0.5 and 2 meters in front of the robot, update the status of the tree
            if self.treatment_status[idx_kd_front] == 0:
                self.treatment_status[idx_kd_front] = 1
                self.qt_window.plotter.update_in_progress_tree(self.test_tree_positions[idx_kd_front][0])
                self.qt_window.console.appendPlainText("Tree " + str(self.test_tree_nums[idx_kd_front][0]) + " started")

        # Publish data about the test tree to the side of the robot
        if len(idx_on_side) > 0:
            test_tree_locations_msg = TestTreeLocationMulti()
            for i in idx_on_side:
                test_tree_location_msg = TestTreeLocation()
                test_tree_location_msg.tree_id = self.test_tree_nums[idx_kd_on_side[i]]
                test_tree_location_msg.position = test_tree_positions_rel_robot[i,0]
                test_tree_locations_msg.trees.append(test_tree_location_msg)
            # Create ros timestamp from float time
            test_tree_locations_msg.header.stamp = time_stamp
            self.test_tree_pub.publish(test_tree_locations_msg)

        # Update the status of trees that are beyond 2 meters behind the robot to completed
        for i in idx_kd_behind:
            if self.treatment_status[i] == 1:
                self.qt_window.console.appendPlainText("Tree " + str(self.test_tree_nums[i]) + " completed")
                self.treatment_status[i] = 2

        # Update the plot with the status of the trees
        complete_tree_positions = self.test_tree_positions[self.treatment_status == 2]
        self.qt_window.plotter.update_complete(complete_tree_positions)

    def plot_clicked(self, x, y, shift_pressed):
        # Method for handling when the plot is clicked, if shift is held down, set the particle start position to the
        # clicked location

        self.plot_x_click = x
        self.plot_y_click = y
        if not self.pf_active:
            # Probably don't need to print to console, but, well i do
            self.qt_window.console.appendPlainText("Plot clicked at: x = " + str(round(x, 2)) + ", y = " + str(round(y, 2)))
            if shift_pressed:
                self.set_particle_start()
            else:
                self.qt_window.console.appendPlainText("Shift click to set particle start position")

    def set_particle_start(self):
        # Method for setting the particle start block position to the clicked location

        if self.plot_x_click is None:
            self.qt_window.console.appendPlainText("No plot click recorded")
            return
        self.qt_window.start_x_input.setText(str(round(self.plot_x_click, 2)))
        self.qt_window.start_y_input.setText(str(round(self.plot_y_click, 2)))
        self.reset_app()

    def adjust_pf_settings(self):
        # Method for adjusting the particle filter settings by opening a settings dialog

        if self.pf_active:
            self.qt_window.console.appendPlainText("Cannot adjust settings while PF is running")
            return

        settings_dialog = SettingsDialog(current_settings=self.pf_settings)

        if settings_dialog.exec_() == QDialog.Accepted:
            new_settings = settings_dialog.get_settings()
            if new_settings is None:
                self.qt_window.console.appendPlainText("Update Failed")
            else:
                self.pf_settings = new_settings
                for key, value in self.pf_settings.items():
                    self.qt_window.console.appendPlainText(key + ": " + str(value))
                self.reset_app()

    def check_end_of_row(self, robot_position, time_stamp):
        row_end_line = np.array([[30.7, 125.74], [55.5, 86.0]])
        row_start_line = np.array([[12.95, 97.0], [4.97, 4.38]])

        def slope_from_points(p1, p2):
            return (p2[1] - p1[1]) / (p2[0] - p1[0])

        def intersection_of_robot_path_with_line(robot_pos, p1, m_line):
            # Find the x intersection of the robot's path with the given line
            tan_theta = np.tan(robot_pos[2])
            x_intersection = (p1[1] - robot_pos[1] + tan_theta * robot_pos[0] - m_line * p1[0]) / (tan_theta - m_line)
            y_intersection = robot_pos[1] + tan_theta * (x_intersection - robot_pos[0])
            return x_intersection, y_intersection

        def distance_between_points(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        m_start = slope_from_points(row_start_line[0], row_start_line[1])
        m_end = slope_from_points(row_end_line[0], row_end_line[1])

        x_start_int, y_start_int = intersection_of_robot_path_with_line(robot_position, row_start_line[0], m_start)
        x_end_int, y_end_int = intersection_of_robot_path_with_line(robot_position, row_end_line[0], m_end)

        dist_to_start = distance_between_points([x_start_int, y_start_int], robot_position[:2])
        dist_to_end = distance_between_points([x_end_int, y_end_int], robot_position[:2])

        print("dist to start: ", dist_to_start)
        print("dist to end: ", dist_to_end)

        # return dist_to_start, dist_to_end

def run_pf_process(pf_update_queue, pf_return_queue, map_data):
    # Method for running the particle filter in a separate process

    # Initialize the particle filter engine
    pf_engine = PFEngine(map_data)

    while True:
        # Get the most recent data from the queue
        odom_data, tree_msg, pf_setup_data = pf_update_queue.get()

        # If the particle filter is being reset, reset the particle filter
        if pf_setup_data is not None:
            pf_engine.reset_pf(pf_setup_data)
            particles = pf_engine.downsample_particles(max_samples=1000)
            best_guess = pf_engine.particles.mean(axis=0)
            num_particles = len(pf_engine.particles)
            converged = False
            time_stamp = None
        else:
            # Otherwise, update the particle filter with the odom data and tree data
            if odom_data is not None:
                x_odom = odom_data['x_odom']
                theta_odom = odom_data['theta_odom']
                time_stamp = odom_data['time_odom']
                num_readings = odom_data['num_readings']
                pf_engine.save_odom(x_odom, theta_odom, time_stamp, num_readings=num_readings)
            if tree_msg is not None:
                pf_engine.scan_update(tree_msg)

            # Get 1000 random particles from the particle filter, the average of the particles, the total number of
            # particles, and whether the particle filter has converged
            particles = pf_engine.downsample_particles(max_samples=1000)
            best_guess = pf_engine.particles.mean(axis=0)
            converged = pf_engine.check_convergence()
            num_particles = len(pf_engine.particles)

            time_stamp = tree_msg['time_stamp']

        # Put the data back into the queue
        pf_return_queue.put((particles, best_guess, converged, num_particles, time_stamp))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    app = QApplication(sys.argv)
    # window = MyMainWindow()
    particle_filter = ParticleFilterBagFiles(app)
    particle_filter.qt_window.show()

    sys.exit(app.exec_())





