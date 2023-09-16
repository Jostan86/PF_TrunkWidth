import csv
import os
import sys


def extract_data_from_csvs(directory):
    # Define the header of the new csv file
    header = ['File name', 'Overall Average Time Converged', 'Overall Average Convergence Rate', 'Benchmark Time',
              'Adjusted Time']

    # Placeholder for the data
    data = []

    filenames = os.listdir(directory)
    filenames.sort()

    # Loop through every file in the directory
    for filename in filenames:
        if filename.endswith(".csv") and filename.startswith("results"):  # Check if the file is a csv
            with open(os.path.join(directory, filename), 'r') as f:
                # Read the csv and get the 22, 23, 24 rows (considering 0-based index)
                lines = f.readlines()[21:24]

                # Extract the desired data from the lines
                overall_avg_time_converged = lines[0].split(",")[1].strip()
                overall_avg_conv_rate = lines[1].split(",")[1].strip()
                benchmark_time = lines[2].split(",")[1].strip()
                try:
                    adjusted_time = float(overall_avg_time_converged) * (5/float(benchmark_time))
                except ZeroDivisionError:
                    print(f"ZeroDivisionError: {filename}")
                    adjusted_time = float(overall_avg_time_converged) * (5/float(12))
                adjusted_time = round(adjusted_time, 2)
                data.append([filename, overall_avg_time_converged, overall_avg_conv_rate, benchmark_time, adjusted_time])

    # Write the data to a new csv file
    with open(os.path.join(directory, 'summary.csv'), 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Data extracted and saved to {os.path.join(directory, 'summary.csv')}")

# Example usage
# extract_data_from_csvs('/path/to/directory')

if __name__=='__main__':
    extract_data_from_csvs('../data/test_results/results_kld3/')