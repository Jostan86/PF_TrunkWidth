#!/usr/bin/env python3

import json
import numpy as np
from pf_engine_og import PFEngineOG
from pf_engine_cpy import PFEngine
import bisect
import time
import scipy
import pandas as pd
import csv
import argparse
import glob
from scipy.ndimage import label
import scipy.spatial

def parse_args():
    parser = argparse.ArgumentParser(description='Analyzes the particle filter system.')

    # Adding an argument
    parser.add_argument('-pd', '--particle_density', type=int, default=500, help='Number of particles per square '
                                                                                  'meter')
    parser.add_argument('-sd_d', '--dist_sd', type=float, default=0.35, help='Distance sensor model standard deviation')
    parser.add_argument('-sd_w', '--width_sd', type=float, default=0.025, help='Width sensor model standard '
                                                                                 'deviation')
    parser.add_argument('-q_dist', '--q_dist', type=float, default=0.4, help='idk')
    parser.add_argument('-q_angle', '--q_angle', type=float, default=10, help='idk')
    parser.add_argument('-r_dist', '--r_dist', type=float, default=0.6, help='idk')
    parser.add_argument('-r_angle', '--r_angle', type=float, default=20, help='idk, in degrees')

    parser.add_argument('-bin_size', '--bin_size', type=float, default=0.2, help='Size of the bins for the kld '
                                                                                 'resampling')
    parser.add_argument('-bin_angle', '--bin_angle', type=float, default=5, help='Size of the bins for the kld '
                                                                                 'resampling in degrees')
    parser.add_argument('-epsilon', '--epsilon', type=float, default=0.05, help='Maximum allowable error in K-L ')
    parser.add_argument('-delta', '--delta', type=float, default=0.05, help='Desired confidence in the calculated ')

    parser.add_argument('-n_min', '--num_particles_min', type=int, default=100, help='Minimum number of particles')
    parser.add_argument('-n_max', '--num_particles_max', type=int, default=2000000, help='Maximum number of particles')

    parser.add_argument('--exclude_width', action='store_true', help='Include to ignore width in weight '
                                                                         'calculation')

    parser.add_argument('-nt', '--num_trials', type=int, default=3, help='Number of trials to run for '
                                                                         'each start position')
    parser.add_argument('--benchmark', action='store_true', help='Include to run the benchmark')

    parser.add_argument('-dir_data', '--directory_data', type=str,
                        default='/media/jostan/MOAD/research_data/2023_orchard_data/pf_data/', help='Directory with '
                                                                                                    'the data')
    parser.add_argument('-dir_out', '--directory_out', type=str, default='/media/jostan/MOAD/research_data'
                                                '/2023_orchard_data/pf_results/', help='Directory to save the output')

    parser.add_argument('--verbose', action='store_true', help='Include to print more information')

    parser.add_argument('-name', '--name', type=str, default='test', help='Name of the trial')

    args = parser.parse_args()

    return args

def get_map_data(args, include_sprinklers=False, separate_test_trees=False, move_origin=True, origin_offset=5):
    # Path to the tree data dictionary
    tree_data_path = args.directory_data + 'tree_list_mod3.json'

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

def get_test_starts(args):

    file_path = args.directory_data + 'test_starts.csv'
    df = pd.read_csv(file_path, header=1)

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

class ParticleFilterEvaluator:
    def __init__(self, args):

        self.args = args
        print(self.args.bin_angle)

        self.results_save_dir = self.args.directory_out
        self.saved_data_dir = self.args.directory_data

        self.benchmark = self.args.benchmark
        self.running_benchmark = False
        self.benchmark_runs = 20

        self.msg_order = []
        self.time_stamps = []
        self.time_stamps_keys = []
        self.cur_data_pos = 0
        self.cur_odom_pos = 0
        self.cur_img_pos = 0

        self.img_data = None
        self.odom_data = None

        self.pf_active = False

        self.start_x = None
        self.start_y = None
        self.start_width = None
        self.start_height = None
        self.start_rotation = None
        self.start_time = None

        self.map_data = get_map_data(self.args)

        self.round_exclude_time = 0

        self.test_starts = get_test_starts(self.args)
        self.num_trials = self.args.num_trials
        self.distances = []
        self.convergences = []
        self.run_times = []
        self.distances_benchmark = []
        self.convergences_benchmark = []
        self.run_times_benchmark = []

        self.run_num = 0
        self.correct_convergence_threshold = 0.5 # meters

        if self.args.exclude_width:
            self.include_width = False
        else:
            self.include_width = True

        self.verbose = self.args.verbose

        self.pf_engine = None

        self.particle_density = self.args.particle_density # particles per square meter

        if self.benchmark:
            self.running_benchmark = True
            self.run_benchmark()
            self.running_benchmark = False
        self.run_trials()


    def run_trials(self):

        if self.verbose:
            print("Starting tests...")

        for start_num in range(len(self.test_starts)):
            run_times = []
            trials_converged = []
            distances = []
            self.load_start_location(start_num)

            if self.verbose:
                print("Starting tests for start location {}...".format(start_num))

            for i in range(self.num_trials):

                if self.verbose:
                    print("Starting trial {}...".format(i + 1))

                self.reset_test()

                round_start_time = time.time()
                self.round_exclude_time = 0

                self.run_pf()

                run_times.append(time.time() - round_start_time - self.round_exclude_time)
                correct_convergence, distance = self.check_converged_distance()
                distances.append(distance)
                trials_converged.append(correct_convergence)


            # print run times
            if self.verbose:
                print("Results for start location {}: ".format(start_num))
                for runtime, distance, convergence in zip(run_times, distances, trials_converged):
                    msg = str(round(runtime, 2)) + "s" + "   Converged: " + str(convergence) + "   Distance: " + \
                          str(round(distance, 2)) + "m"

                    print(msg)

            self.distances.append(distances)
            self.convergences.append(trials_converged)
            self.run_times.append(run_times)


        self.process_results()

    def run_pf(self):
        self.pf_active = True
        while self.pf_active:
            self.send_next_msg()
            if not self.pf_active:
                break

            if (self.pf_engine.histogram is not None and self.img_data[self.cur_img_pos] is not None and
                    self.msg_order[self.cur_data_pos] == 1):
                start_time_dist = time.time()
                correct_convergence = self.check_convergence(self.pf_engine.histogram)
                if correct_convergence:
                    self.pf_active = False
                self.round_exclude_time += time.time() - start_time_dist
    def run_pf_benchmark(self):
        self.pf_active = True
        while self.pf_active:
            self.send_next_msg()
            if not self.pf_active:
                break
        if self.pf_engine.particles.shape[0] == 100 and self.img_data[self.cur_img_pos] is not None:

            start_time_dist = time.time()
            # Find the distance between every pair of particles
            dists = scipy.spatial.distance.pdist(self.pf_engine.particles)
            # Find the maximum distance between any pair of particles
            max_dist = np.max(dists)
            self.round_exclude_time += time.time() - start_time_dist
            if max_dist < 1:
                self.pf_active = False
            else:
                self.pf_active = False

    def send_next_msg(self):

        if self.cur_data_pos >= len(self.msg_order):
            if self.verbose:
                print("reached end of run")
            self.cur_img_pos -= 1
            self.pf_active = False
            return

        if self.msg_order[self.cur_data_pos] == 0:
            x_odom = self.odom_data[self.cur_odom_pos]['x_odom']
            theta_odom = self.odom_data[self.cur_odom_pos]['theta_odom']
            time_stamp_odom = self.odom_data[self.cur_odom_pos]['time_stamp']
            if self.running_benchmark:
                self.pf_engine.save_odom_loaded(x_odom, theta_odom, time_stamp_odom)
            else:
                self.pf_engine.save_odom(x_odom, theta_odom, time_stamp_odom)

            self.cur_odom_pos += 1
            self.cur_data_pos += 1

        elif self.msg_order[self.cur_data_pos] == 1:
            img_data = self.img_data[self.cur_img_pos]
            self.cur_img_pos += 1
            self.cur_data_pos += 1

            if img_data is not None:
                # return

                tree_positions = np.array(img_data['tree_data']['positions'])
                widths = np.array(img_data['tree_data']['widths'])
                classes = np.array(img_data['tree_data']['classes'])

                tree_data = {'positions': tree_positions, 'widths': widths, 'classes': classes}
                if self.running_benchmark:
                    self.pf_engine.save_scan(tree_data)
                else:
                    self.pf_engine.scan_update(tree_data)
            elif not self.running_benchmark:
                self.pf_engine.resample_particles()
    def process_results(self):
        convergences = np.array(self.convergences)
        run_times = np.array(self.run_times)

        # Could also be measureing how many images it had to look at, although i guess that doesn't really tell me time

        # Maybe the benchmark idea is best, but idk about that either

        # can also track number of particles over time, could maybe somehow combine that with the other ideas

        # Here's what I'll do. I just had GPT make me a benchmark script which I'll run and report, I'll also make a
        # seperate script that runs the pf for a standard start location maybe like 30 times and reports the average
        # time to convergence and correct convergence rate.

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


        if self.args.name == "test":
            # determine which number to give results file by
            results_files = glob.glob((self.results_save_dir + "results*.csv"))
            results_files = [file.split('/')[-1] for file in results_files]
            results_files = [int(file[7:-4]) for file in results_files]
            if len(results_files) == 0:
                results_file_num = 0
            else:
                results_file_num = max(results_files) + 1

            results_file_path = self.results_save_dir + "results{}.csv".format(results_file_num)
            benchmark_file_path = self.results_save_dir + "benchmarks/benchmark{}.csv".format(results_file_num)
        else:
            results_file_path = self.results_save_dir + "results_{}.csv".format(self.args.name)
            benchmark_file_path = self.results_save_dir + "benchmarks/benchmark_{}.csv".format(self.args.name)

        # Save the results to a csv file
        with open(results_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Start Location", "Convergence Rate", "Average Time", "Average Time Converged"])
            for i in range(len(self.test_starts)):
                writer.writerow([i, convergence_rates[i], avg_times[i], avg_times_converged[i]])

            writer.writerow(["Overall Average Time Converged", overall_avg_time_converged])
            writer.writerow(["Overall Average Convergence Rate", overall_avg_convergence_rate])
            if self.benchmark:
                writer.writerow(["Benchmark Time", round(np.mean(self.run_times_benchmark), 2)])

        if self.benchmark:

            with open(benchmark_file_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Runtime", "Distance", "Convergence"])
                for runtime, distance, convergence in zip(self.run_times_benchmark, self.distances_benchmark, self.convergences_benchmark):
                    writer.writerow([round(runtime, 2), distance, convergence])
                writer.writerow(["Average Runtime", round(np.mean(self.run_times_benchmark), 2)])
                writer.writerow(["Average Distance", round(np.mean(self.distances_benchmark), 2)])
                writer.writerow(["Average Convergence", round(np.mean(self.convergences_benchmark), 2)])

        # Print the results
        if self.verbose:
            print("Results:")
            for i in range(len(self.test_starts)):
                print("Start Location: {}   Convergence Rate: {}   Average Time: {}   Average Time Converged: {}"
                      .format(i, convergence_rates[i], avg_times[i], avg_times_converged[i]))
            print("Overall Average Time Converged: {}".format(overall_avg_time_converged))
            print("Overall Average Convergence Rate: {}".format(overall_avg_convergence_rate))

    def load_start_location(self, start_num):

        test_info = self.test_starts[start_num]

        run_num = test_info["run_num"]

        # open the data file
        self.open_saved_data(run_num)

        # set the bag time to the correct time
        self.start_time = test_info["start_time"]

        # set the start pose center to the correct center
        self.start_x = test_info["start_pose_center"][0]
        self.start_y = test_info["start_pose_center"][1]
        self.start_width = test_info["start_pose_width"]
        self.start_height = test_info["start_pose_height"]
        self.start_rotation = test_info["start_pose_rotation"]

    def reset_test(self):

        self.set_time_line_location(self.start_time)

        num_particles = int(self.particle_density * self.start_height * self.start_width)

        self.pf_engine = PFEngine(self.map_data, start_pose_center=[self.start_x, self.start_y],
                                  start_pose_height=self.start_height, start_pose_width=self.start_width,
                                  num_particles=num_particles, rotation=np.deg2rad(self.start_rotation), )

        self.pf_engine.include_width = self.include_width

        # Setup the covariances
        self.pf_engine.Q = np.diag([self.args.q_dist, np.deg2rad(self.args.q_angle)]) ** 2
        self.pf_engine.R = np.diag([self.args.r_dist, np.deg2rad(self.args.r_angle)]) ** 2
        self.pf_engine.dist_sd = self.args.dist_sd
        self.pf_engine.width_sd = self.args.width_sd

        self.pf_engine.epsilon = self.args.epsilon
        self.pf_engine.delta = self.args.delta
        self.pf_engine.bin_size = self.args.bin_size
        self.pf_engine.bin_angle = np.deg2rad(self.args.bin_angle)

        self.pf_engine.min_num_particles = self.args.num_particles_min
        self.pf_engine.max_num_particles = self.args.num_particles_max

    def set_time_line_location(self, time_stamp):

        # Find the position of the time stamp in the list of time stamps
        time_stamp_pos = bisect.bisect_left(self.time_stamps, time_stamp)

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



    def open_saved_data(self, run_num):
        self.time_stamps = []
        self.msg_order = []
        self.cur_odom_pos = 0
        self.cur_img_pos = 0
        self.cur_data_pos = 0
        self.img_data = []
        self.odom_data = []
        t_start = None

        data_file_path = self.saved_data_dir + "run_" + str(int(run_num)) + "_gt_data.json"

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


    def check_converged_distance(self):
        current_position = self.pf_engine.best_particle[0:2]
        actual_position_x = self.img_data[self.cur_img_pos]['location_estimate']['x']
        actual_position_y = self.img_data[self.cur_img_pos]['location_estimate']['y']
        actual_position = np.array([actual_position_x, actual_position_y])
        distance = np.linalg.norm(current_position - actual_position)
        if distance < self.correct_convergence_threshold:
            return True, distance
        else:
            return False, distance

    def run_benchmark(self):
        if self.verbose:
            print("Starting benchmark...")

        start_num = 4
        self.run_times_benchmark = []
        self.convergences_benchmark = []
        self.distances_benchmark = []
        self.load_start_location(start_num)

        for i in range(self.benchmark_runs):
            if self.verbose:
                print("Starting run {}...".format(i))

            self.reset_test_bench(i)

            round_start_time = time.time()
            self.round_exclude_time = 0

            self.run_pf_benchmark()

            self.run_times_benchmark.append(time.time() - round_start_time - self.round_exclude_time)
            correct_convergence, distance = self.check_converged_distance()
            self.distances_benchmark.append(distance)
            self.convergences_benchmark.append(correct_convergence)



        # print run times
        if self.verbose:
            print("Results for start location {}: ".format(start_num))
            for runtime, distance, convergence in zip(self.run_times_benchmark, self.distances_benchmark, self.convergences_benchmark):
                msg = str(round(runtime, 2)) + "s" + "   Converged: " + str(convergence) + "   Distance: " + \
                      str(round(distance, 2)) + "m"

                print(msg)
            print("Average time: " + str(round(np.mean(self.run_times_benchmark), 2)) + "s")
            print("Converged: " + str(sum(self.convergences_benchmark)) + "/" + str(len(self.convergences_benchmark)))

    def reset_test_bench(self, iteration):

        self.set_time_line_location(self.start_time)

        particle_density = 500
        num_particles = int(particle_density * self.start_height * self.start_width)

        self.pf_engine = PFEngineOG(self.map_data, start_pose_center=[self.start_x, self.start_y],
                                  start_pose_height=self.start_height, start_pose_width=self.start_width,
                                  num_particles=num_particles, rotation=np.deg2rad(self.start_rotation), rand_seed=iteration)

        self.pf_engine.include_width = True

        # Setup the covariances
        self.pf_engine.Q = np.diag([0.4, np.deg2rad(10.0)]) ** 2
        self.pf_engine.R = np.diag([.6, np.deg2rad(20.0)]) ** 2
        self.pf_engine.dist_sd = 0.35
        self.pf_engine.width_sd = 0.025

        self.pf_engine.min_num_particles = 100
        self.pf_engine.max_num_particles = 2000000

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
            if self.verbose:
                print("Feature Sizes: ", feature_sizes)
                print("Feature Size Max: ", feature_size_max)
            if feature_sizes[0] < feature_size_max:
                return True
            else:
                return False
        else:
            return False



if __name__ == "__main__":
    args = parse_args()
    pf_evaluator = ParticleFilterEvaluator(args)
    # pf_evaluator.run_benchmark()
    # pf_evaluator.run_trials()






