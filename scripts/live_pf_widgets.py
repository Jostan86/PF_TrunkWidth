from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QDialog, QPlainTextEdit, QCheckBox, QDialogButtonBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np

class ClickablePlotWidget(pg.PlotWidget):
    # This class is for a plot widget that emits a signal when clicked about where it was clicked. It also distinguishes
    # between a normal click and a shift-click

    # Define a custom signal
    clicked = pyqtSignal(float, float, bool)  # Signal to emit x and y coordinates

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

class SettingsDialog(QDialog):
    # This class is for a dialog box that allows the user to change settings for the particle filter
    def __init__(self, current_settings=None, parent=None, ):
        super(SettingsDialog, self).__init__(parent)
        self.init_ui(current_settings)

    def init_ui(self, current_settings):
        layout = QVBoxLayout()

        # Movement noise (linear)
        self.linear_noise_label = QLabel('Movement noise (linear):')
        self.linear_noise_edit = QLineEdit(str(current_settings['r_dist']))
        linear_layout = QHBoxLayout()
        linear_layout.addWidget(self.linear_noise_label)
        linear_layout.addWidget(self.linear_noise_edit)
        layout.addLayout(linear_layout)

        # Movement noise (angular)
        self.angular_noise_label = QLabel('Movement noise (angular):')
        self.angular_noise_edit = QLineEdit(str(current_settings['r_angle']))
        angular_layout = QHBoxLayout()
        angular_layout.addWidget(self.angular_noise_label)
        angular_layout.addWidget(self.angular_noise_edit)
        layout.addLayout(angular_layout)

        # width sensor standard deviation
        self.width_sensor_label = QLabel('Width sensor standard deviation:')
        self.width_sensor_edit = QLineEdit(str(current_settings['width_sd']))
        width_layout = QHBoxLayout()
        width_layout.addWidget(self.width_sensor_label)
        width_layout.addWidget(self.width_sensor_edit)
        layout.addLayout(width_layout)

        # Tree position sensor std dev
        self.tree_position_label = QLabel('Tree position sensor std dev:')
        self.tree_position_edit = QLineEdit(str(current_settings['dist_sd']))
        tree_position_layout = QHBoxLayout()
        tree_position_layout.addWidget(self.tree_position_label)
        tree_position_layout.addWidget(self.tree_position_edit)
        layout.addLayout(tree_position_layout)

        # Epsilon
        self.epsilon_label = QLabel('Epsilon:')
        self.epsilon_edit = QLineEdit(str(current_settings['epsilon']))
        epsilon_layout = QHBoxLayout()
        epsilon_layout.addWidget(self.epsilon_label)
        epsilon_layout.addWidget(self.epsilon_edit)
        layout.addLayout(epsilon_layout)

        # Delta
        self.delta_label = QLabel('Delta:')
        self.delta_edit = QLineEdit(str(current_settings['delta']))
        delta_layout = QHBoxLayout()
        delta_layout.addWidget(self.delta_label)
        delta_layout.addWidget(self.delta_edit)
        layout.addLayout(delta_layout)

        # Bin size
        self.bin_size_label = QLabel('Bin size:')
        self.bin_size_edit = QLineEdit(str(current_settings['bin_size']))
        bin_size_layout = QHBoxLayout()
        bin_size_layout.addWidget(self.bin_size_label)
        bin_size_layout.addWidget(self.bin_size_edit)
        layout.addLayout(bin_size_layout)

        # Bin angle
        self.bin_angle_label = QLabel('Bin angle:')
        self.bin_angle_edit = QLineEdit(str(current_settings['bin_angle']))
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

        # # Hard-coded row number positions
        # row_num_xs = [4.9, 5.5, 6.05, 6.65, 7.4, 7.45, 7.9, 8.65, 9.2, 9.65, 10.25, 10.65, 11.05, 11.6, 12.1, 12.65,
        #               13.2]
        # row_num_ys = [4.9, 10.8, 16.5, 22.35, 28.1, 33.3, 39.05, 45.05, 51.05, 56.5, 62.6, 68.25, 73.9, 79.55, 85.6,
        #               91.5, 97.3]
        # row_nums = [i for i in range(len(row_num_xs))]

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


        # Make plot items for things that will be updated
        self.particle_plot_item = self.plot_widget.plot([], [], pen=None, symbol='o',
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
            self.particle_plot_item.setData([], [])


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

