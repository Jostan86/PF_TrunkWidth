import numpy as np


def check_end_of_row(robot_position):
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

    return dist_to_start, dist_to_end


# Testing the function
robot_pos = np.array([20, 40, np.deg2rad(58)])
dist_to_start, dist_to_end = check_end_of_row(robot_pos)
print(f"Distance to Start: {dist_to_start}")
print(f"Distance to End: {dist_to_end}")
