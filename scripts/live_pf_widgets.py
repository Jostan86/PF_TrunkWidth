from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QDialog, QPlainTextEdit, QCheckBox, QDialogButtonBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np



class SettingsDialog(QDialog):
    # This class is for a dialog box that allows the user to change settings for the particle filter
    def __init__(self, current_settings=None, parent=None, ):
        super(SettingsDialog, self).__init__(parent)
        self.init_ui(current_settings)

    def init_ui(self, current_settings):
        layout = QVBoxLayout()

        # Movement noise (linear)
        self.linear_noise_label = QLabel('Movement noise - linear (m/s):')
        self.linear_noise_edit = QLineEdit(str(current_settings['r_dist']))
        tool_tip = ('The standard deviation of the noise added to the linear velocity of the robot')
        self.linear_noise_edit.setToolTip(tool_tip)
        self.linear_noise_label.setToolTip(tool_tip)
        linear_layout = QHBoxLayout()
        linear_layout.addWidget(self.linear_noise_label)
        linear_layout.addWidget(self.linear_noise_edit)
        layout.addLayout(linear_layout)

        # Movement noise (angular)
        self.angular_noise_label = QLabel('Movement noise - angular (deg/s):')
        self.angular_noise_edit = QLineEdit(str(current_settings['r_angle']))
        tool_tip = ('The standard deviation of the noise added to the angular velocity of the robot.')
        self.angular_noise_edit.setToolTip(tool_tip)
        self.angular_noise_label.setToolTip(tool_tip)
        angular_layout = QHBoxLayout()
        angular_layout.addWidget(self.angular_noise_label)
        angular_layout.addWidget(self.angular_noise_edit)
        layout.addLayout(angular_layout)

        # width sensor standard deviation
        self.width_sensor_label = QLabel('Width sensor std dev (m):')
        self.width_sensor_edit = QLineEdit(str(current_settings['width_sd']))
        tool_tip = "The expected standard deviation of the width sensor."
        self.width_sensor_edit.setToolTip(tool_tip)
        self.width_sensor_label.setToolTip(tool_tip)
        width_layout = QHBoxLayout()
        width_layout.addWidget(self.width_sensor_label)
        width_layout.addWidget(self.width_sensor_edit)
        layout.addLayout(width_layout)

        # Tree position sensor std dev
        self.tree_position_label = QLabel('Tree position sensor std dev (m):')
        self.tree_position_edit = QLineEdit(str(current_settings['dist_sd']))
        tool_tip = "The expected standard deviation of the tree position sensor."
        self.tree_position_edit.setToolTip(tool_tip)
        self.tree_position_label.setToolTip(tool_tip)
        tree_position_layout = QHBoxLayout()
        tree_position_layout.addWidget(self.tree_position_label)
        tree_position_layout.addWidget(self.tree_position_edit)
        layout.addLayout(tree_position_layout)

        # Epsilon
        self.epsilon_label = QLabel('Epsilon:')
        self.epsilon_edit = QLineEdit(str(current_settings['epsilon']))
        tool_tip = ("The allowable error for the KLD sampling algorithm. A smaller value will result in more "
                    "particles being generated.")
        self.epsilon_edit.setToolTip(tool_tip)
        self.epsilon_label.setToolTip(tool_tip)
        epsilon_layout = QHBoxLayout()
        epsilon_layout.addWidget(self.epsilon_label)
        epsilon_layout.addWidget(self.epsilon_edit)
        layout.addLayout(epsilon_layout)

        # Delta
        self.delta_label = QLabel('Delta:')
        self.delta_edit = QLineEdit(str(current_settings['delta']))
        tool_tip = ("Another term for the kld sampling. A smaller value will result in more particles being generated. "
                    "I generally have not changed this value, and use epsilon instead.")
        self.delta_edit.setToolTip(tool_tip)
        self.delta_label.setToolTip(tool_tip)
        delta_layout = QHBoxLayout()
        delta_layout.addWidget(self.delta_label)
        delta_layout.addWidget(self.delta_edit)
        layout.addLayout(delta_layout)

        # Bin size
        self.bin_size_label = QLabel('Bin size (m):')
        self.bin_size_edit = QLineEdit(str(current_settings['bin_size']))
        tool_tip = ("The size of the bins used to discretize the map. A smaller value will result in more particles "
                    "being generated, but also more computation time, adjusting epsilon is usually a better "
                    "option.")
        self.bin_size_edit.setToolTip(tool_tip)
        self.bin_size_label.setToolTip(tool_tip)
        bin_size_layout = QHBoxLayout()
        bin_size_layout.addWidget(self.bin_size_label)
        bin_size_layout.addWidget(self.bin_size_edit)
        layout.addLayout(bin_size_layout)

        # Bin angle
        self.bin_angle_label = QLabel('Bin angle (deg):')
        self.bin_angle_edit = QLineEdit(str(current_settings['bin_angle']))
        tool_tip = ("The angular size of the bins used to discretize the map. A smaller value will result in more "
                    "particles being generated, but also more computation time, adjusting epsilon is usually a "
                    "better option.")
        bin_angle_layout = QHBoxLayout()
        bin_angle_layout.addWidget(self.bin_angle_label)
        bin_angle_layout.addWidget(self.bin_angle_edit)
        layout.addLayout(bin_angle_layout)

        # Add OK and Cancel buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        self.setWindowTitle("Settings")


    def get_settings(self):
        # Method to retrieve all settings data as a dictionary
        try:
            return {
                'r_dist': float(self.linear_noise_edit.text()),
                'r_angle': float(self.angular_noise_edit.text()),
                'width_sd': float(self.width_sensor_edit.text()),
                'dist_sd': float(self.tree_position_edit.text()),
                'epsilon': float(self.epsilon_edit.text()),
                'delta': float(self.delta_edit.text()),
                'bin_size': float(self.bin_size_edit.text()),
                'bin_angle': float(self.bin_angle_edit.text())
            }
        except ValueError:
            return None

class ClickablePlotWidget(pg.PlotWidget):
    # This class is for a plot widget that emits a signal when clicked about where it was clicked. It also distinguishes
    # between a normal click and a shift-click

    # Define a custom signal
    clicked = pyqtSignal(float, float, bool)  # Signal to emit x and y coordinates

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_repainting = False

    def mousePressEvent(self, event):
        mouse_point = self.plotItem.vb.mapSceneToView(event.pos())
        x = mouse_point.x()
        y = mouse_point.y()

        # Check if Shift key is pressed
        if event.modifiers() & Qt.ShiftModifier:
            self.clicked.emit(x, y, True)
        else:
            self.clicked.emit(x, y, False)

        super().mousePressEvent(event)

    def paintEvent(self, event):
        self.is_repainting = True
        super().paintEvent(event)
        self.is_repainting = False

class ParticleMapPlotter(QMainWindow):
    # Class to handle all the plotting for the particle filter app
    def __init__(self, map_data):
        super().__init__()

        # Indicates whether or not to show tree numbers on the plot
        self.show_nums = False

        # Create a pyqtgraph plot widget with some added functionality
        self.plot_widget = ClickablePlotWidget()

        # Set background to white and lock the aspect ratio
        self.plot_widget.setAspectLocked(True, ratio=1)
        self.plot_widget.setBackground('w')

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

        # Draw the map
        self.draw_plot()

    def draw_plot(self, particles=None):
        """Method to draw the map on the plot widget"""

        # Hard-coded row number positions
        row_num_xs = [4.9, 5.5, 6.05, 6.65, 7.4, 7.45, 7.9, 8.65, 9.2, 9.65, 10.25, 10.65, 11.05, 11.6, 12.1, 12.65,
                      13.2]
        row_num_xs = [x - .75 for x in row_num_xs]
        row_num_ys = [4.9, 10.8, 16.5, 22.35, 28.1, 33.3, 39.05, 45.05, 51.05, 56.5, 62.6, 68.25, 73.9, 79.55, 85.6,
                      91.5, 97.3]
        row_num_ys = [y - 1 for y in row_num_ys]
        row_nums = [96+i for i in range(len(row_num_xs))]

        # Clear the plot widget
        self.plot_widget.clear()

        # Get the positions of the trees, posts, and test trees
        tree_positions = self.positions[self.classes == 0]
        post_positions = self.positions[self.classes == 1]
        test_tree_positions = self.positions[self.classes == 3]

        # Set size of dots
        self.dot_size = 10

        # Add data to the plot widget
        self.plot_widget.plot(tree_positions[:, 0], tree_positions[:, 1], pen=None, symbol='o', symbolBrush='g',
                              symbolSize=self.dot_size, name='Trees')
        self.plot_widget.plot(post_positions[:, 0], post_positions[:, 1], pen=None, symbol='o', symbolBrush='r',
                              symbolSize=self.dot_size, name='Posts')
        self.plot_widget.plot(test_tree_positions[:, 0], test_tree_positions[:, 1], pen=None, symbol='o',
                              symbolBrush=(0, 81, 180), symbolSize=self.dot_size, name='Test Trees')

        # Add numbers to the trees if show_nums is True, which is toggled by a button in the app
        if self.show_nums:
            for i, (x, y) in enumerate(self.positions):
                # tree_num_text = pg.TextItem(
                #     html='<div style="text-align: center"><span style="color: #000000; font-size: 8pt;">{}</span></div>'.format(
                #             self.tree_numbers[i]), anchor=(1.1, 0.5))
                # tree_num_text.setPos(x, y)
                # self.plot_widget.addItem(tree_num_text)

                # Add test tree numbers
                if self.classes[i] == 3:
                    tree_num_text = pg.TextItem(
                        html = '<div style="text-align: center"><span style="color: #000000; font-size: 8pt;">{}</span></div>'.format(
                            self.test_tree_nums[i]), anchor = (-0.1, 0.5))
                    tree_num_text.setPos(x, y)
                    self.plot_widget.addItem(tree_num_text)

            # Add row numbers
            for i, (x, y) in enumerate(zip(row_num_xs, row_num_ys)):
                row_num_text = pg.TextItem(
                    html='<div style="text-align: center"><span style="color: #000000; font-size: 15pt;">{}</span></div>'.format(
                        row_nums[i]), anchor=(0.5, 0.5))
                row_num_text.setPos(x, y)
                self.plot_widget.addItem(row_num_text)





        # Make plot items for things that will be updated
        particles1 = np.zeros(1000)
        self.particle_plot_item = self.plot_widget.plot(particles1, particles1, pen=None, symbol='o',
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
        # Method to update the particles on the plot
        if particles is not None:
            self.particle_plot_item.setData(particles[:, 0], particles[:, 1])
        else:
            # Put 1000 particles in the corner if there are no particles
            particles = np.zeros(1000)
            self.particle_plot_item.setData(particles, particles)


    def update_best_guess(self, best_guess_position):
        # Method to update the best guess on the plot
        if best_guess_position is not None:
            self.best_guess_plot_item.setData([best_guess_position[0]], [best_guess_position[1]])
        else:
            self.best_guess_plot_item.setData([], [])

    def update_in_progress_tree(self, in_progress_tree_position):
        # Method to update the in progress tree on the plot
        if in_progress_tree_position is not None:
            self.in_progress_tree_plot_item.setData([in_progress_tree_position[0]], [in_progress_tree_position[1]])
        else:
            self.in_progress_tree_plot_item.setData([], [])

    def update_complete(self, complete_position):
        # Method to update the test trees which have been treated on the plot
        if complete_position is None:
            self.complete_plot_item.setData([], [])
        elif len(complete_position) == 1:
            complete_position = complete_position[0]
            self.complete_plot_item.setData([complete_position[0]], [complete_position[1]])
        else:
            self.complete_plot_item.setData(complete_position[:, 0], complete_position[:, 1])

