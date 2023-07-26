import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QSlider
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle("My PyQt App")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Left-side widgets
        self.reset_button = QPushButton("Reset")
        self.start_x_input = QLineEdit()
        self.start_y_input = QLineEdit()
        self.radius_input = QLineEdit()
        self.direction_start_slider = QSlider(Qt.Horizontal)
        self.direction_end_slider = QSlider(Qt.Horizontal)
        self.num_particles_input = QLineEdit()
        self.start_stop_button = QPushButton("Start/Stop")
        self.picture_label = QLabel()
        self.current_file_label = QLabel("Current File Name: ")
        self.widths_found_label = QLabel("Widths Found: ")

        # Right-side plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Setup start location layout
        start_location_layout = QHBoxLayout()
        start_location_layout.addWidget(QLabel("Start:"))
        start_location_layout.addWidget(QLabel("X:"))
        start_location_layout.addWidget(self.start_x_input)
        start_location_layout.addWidget(QLabel("Y:"))
        start_location_layout.addWidget(self.start_y_input)

        # Setup radius input layout
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        radius_layout.addWidget(self.radius_input)

        # Setup direction layout
        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("Direction:"))
        direction_layout.addWidget(self.direction_start_slider)
        direction_layout.addWidget(self.direction_end_slider)

        # Setup num particles layout
        num_particles_layout = QHBoxLayout()
        num_particles_layout.addWidget(QLabel("Num Particles:"))
        num_particles_layout.addWidget(self.num_particles_input)

        # Set up layouts
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.reset_button)
        left_layout.addLayout(start_location_layout)
        left_layout.addLayout(radius_layout)
        left_layout.addLayout(direction_layout)
        left_layout.addLayout(num_particles_layout)
        left_layout.addWidget(self.start_stop_button)
        left_layout.addWidget(self.picture_label)
        left_layout.addWidget(self.current_file_label)
        left_layout.addWidget(self.widths_found_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        central_widget.setLayout(main_layout)


def main():
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()