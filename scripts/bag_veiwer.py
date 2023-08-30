#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
import os
import rosbag
from width_estimation import TrunkAnalyzer
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QComboBox, QPlainTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
import bisect
import time

# To run in pycharm open pycharm from the terminal

# To debug in pycharm add the following environment variable:
# LD_PRELOAD: Set it to /usr/lib/x86_64-linux-gnu/libstdc++.so.6


class MyMainWindow(QMainWindow):
    def __init__(self):
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


        self.picture_label = QLabel(self)
        self.picture_label.resize(960, 720)
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



        # Set up layouts
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.reset_button)
        main_layout.addWidget(self.picture_label)
        main_layout.addWidget(self.img_number_label)
        main_layout.addLayout(img_browsing_buttons_layout)
        main_layout.addLayout(data_file_selector_layout)
        main_layout.addLayout(bag_time_line_layout)
        main_layout.addWidget(self.console)
        main_layout.addWidget(self.clear_console_button)


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
    def __init__(self, app):
        self.bridge = CvBridge()

        self.qt_window = MyMainWindow()
        self.qt_app = app

        self.bag_file_dir = "/media/jostan/MOAD/research_data/"
        # self.bag_file_dir = "/media/jostan/MOAD/research_data/hort_farm_data/"
        # self.img_base_dir = "/media/jostan/MOAD/research_data/roza_imgs/"

        file_names = os.listdir(self.bag_file_dir)
        self.bag_file_names = [f for f in file_names if f.endswith(".bag")]
        self.bag_file_names.sort()

        for bag_file_name in self.bag_file_names:
            self.qt_window.data_file_selector.addItem(bag_file_name)

        self.data_file_names = []

        self.paired_imgs = []
        self.time_stamps = []
        self.cur_img_pos = 0
        self.cur_bag_file_name = None

        self.trunk_analyzer = TrunkAnalyzer()

        self.rgb_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"

        self.qt_window.reset_button.clicked.connect(self.reset_app)
        self.qt_window.data_file_open_button.clicked.connect(self.open_bag_file_button_clicked)
        self.qt_window.prev_img_button.clicked.connect(self.prev_button_clicked)
        self.qt_window.next_img_button.clicked.connect(self.next_button_clicked)
        self.qt_window.play_fwd_button.clicked.connect(self.play_button_clicked)

        self.qt_window.bag_time_line.returnPressed.connect(self.time_stamp_line_edited)

        self.save_data = True

        self.time_stamps_keys = []
        self.cur_data_pos = 0
        self.reset_app()




    def reset_app(self):
        ...

    def update_img_label(self):
        self.qt_window.img_number_label.setText("Image: " + str(self.cur_img_pos + 1) + "/" + str(len(self.paired_imgs)))

    def get_trunk_data(self):
        # try:
        tree_positions, widths, classes, img_seg = self.trunk_analyzer.pf_helper(
            self.paired_imgs[self.cur_img_pos][1],
            self.paired_imgs[self.cur_img_pos][0],
            show_seg=True)
        # except IndexError:
        #     print("Index error")

        if widths is not None:
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

        # if self.save_data:
        #     timestamp = str(int(1000 * self.time_stamps_img[self.cur_img_pos]))
        #     # Save the image to file
        #     file_name = self.img_base_dir + "run_" + str(self.run_num) + "/" + timestamp + ".png"
        #     cv2.imwrite(file_name, img_seg)

        return tree_positions, widths, classes

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


        self.cur_img_pos -= 1
        if self.cur_img_pos <= 0:
            self.cur_img_pos = 0
            self.qt_window.console.appendPlainText("Reached the beginning of the bag file / loaded data")
            # return
        else:
            # self.qt_window.load_image(self.paired_imgs[self.cur_img_pos][1])
            self.get_segmentation()
            self.qt_window.bag_time_line.setText(str(self.time_stamps[self.cur_img_pos]))
        self.update_img_label()

    def play_button_clicked(self):
        if self.qt_window.play_fwd_button.text() == "Play":
            self.qt_window.play_fwd_button.setText("Stop")
            while self.cur_img_pos < len(self.paired_imgs) - 1 and self.qt_window.play_fwd_button.text() == "Stop":
                self.next_button_clicked()
                time.sleep(0.02)
            self.qt_window.play_fwd_button.setText("Play")
        else:
            self.qt_window.play_fwd_button.setText("Play")


    def next_button_clicked(self):
        self.cur_img_pos += 1
        if self.cur_img_pos >= len(self.paired_imgs) - 1:
            self.cur_img_pos = len(self.paired_imgs) - 1
            self.qt_window.console.appendPlainText("Reached end of the bag file")
            # return
        else:
            # self.qt_window.load_image(self.paired_imgs[self.cur_img_pos][1])
            self.get_segmentation()
            self.qt_window.bag_time_line.setText(str(self.time_stamps[self.cur_img_pos]))
        self.update_img_label()

    def get_segmentation(self):
        _, _, _, = self.get_trunk_data()


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
        self.cur_img_pos = time_stamp_pos

        self.qt_window.load_image(self.paired_imgs[self.cur_img_pos][1])

    def open_bag_file_button_clicked(self):
        bag_file_name = self.qt_window.data_file_selector.currentText()
        self.open_bag_file(bag_file_name, from_button=True)

    def open_bag_file(self, bag_file_name, from_button=False):

        self.cur_bag_file_name = bag_file_name

        if not from_button:
            self.qt_window.data_file_selector.setCurrentText(self.cur_bag_file_name)

        # Change the button text
        self.qt_window.data_file_open_button.setText("---")

        # Make sure the change is visible
        self.qt_window.data_file_open_button.repaint()

        path = self.bag_file_dir + self.cur_bag_file_name
        bag_data = rosbag.Bag(path)
        self.paired_imgs = []
        self.time_stamps = []
        self.cur_img_pos = 0
        depth_msgs = []
        color_msgs = []

        t_start = None
        for topic, msg, t in bag_data.read_messages(topics=[self.rgb_topic, self.depth_topic]):
            if t_start is None:
                t_start = t.to_sec()
            if topic == self.depth_topic:
                depth_msg = msg
                depth_msgs.append(depth_msg)
                # paired_img = pair_messages(depth_msg, color_msg)
                # if paired_img is not None:
                #     self.paired_imgs.append(paired_img)
                #     self.time_stamps.append(t.to_sec() - t_start)

            elif topic == self.rgb_topic:
                color_msg = msg
                color_msgs.append(color_msg)
                # paired_img = pair_messages(depth_msg, color_msg)
                # if paired_img is not None:
                #     self.paired_imgs.append(paired_img)
                #     self.time_stamps.append(t.to_sec() - t_start)

        # Pair messages by comparing thier time stamps
        for i in range(len(depth_msgs)):
            depth_time = depth_msgs[i].header.stamp.to_sec()
            for j in range(len(color_msgs)):
                color_time = color_msgs[j].header.stamp.to_sec()
                if abs(depth_time - color_time) < 0.01:
                    try:
                        depth_image = self.bridge.imgmsg_to_cv2(depth_msgs[i], "passthrough")
                        color_img = self.bridge.imgmsg_to_cv2(color_msgs[j], "bgr8")
                    except CvBridgeError as e:
                        print(e)

                    # # remove 16 pixels from the top edge of each of the images to size them for yolo
                    # depth_image = depth_image[16:, :]
                    # color_img = color_img[16:, :]

                    self.paired_imgs.append([depth_image, color_img])
                    self.time_stamps.append(depth_msgs[i].header.stamp.to_sec())
                    break

        self.qt_window.data_file_open_button.setText("Open")

        if from_button:
            self.qt_window.bag_time_line.setText(str(self.time_stamps[0]))
            self.time_stamp_line_edited()
            self.get_segmentation()

        # Write a message to the console
        self.qt_window.console.appendPlainText("Opened bag file: " + self.cur_bag_file_name)
        self.qt_window.console.appendPlainText("Number of images: " + str(len(self.paired_imgs)))





if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = MyMainWindow()
    particle_filter = ParticleFilterBagFiles(app)
    particle_filter.qt_window.show()

    sys.exit(app.exec_())





