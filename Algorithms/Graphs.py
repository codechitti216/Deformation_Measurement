import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = int(input("Ask? : "))

values = []

def euclidean_distance(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

def contour_list(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_name = 0
    frame_points = []
    for idx, contour in enumerate(contours):
        M = cv2.moments(contour)
        contour_name += 1
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)
            cv2.putText(frame, f"{contour_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frame_points.append([contour_name, [cX, cY]])
    return frame_points

def conversion(data, value):
    time_stamps = data[0]
    positions = data[1]
    output_positions = [positions[0]]
    output_time = [time_stamps[0]]

    for i in range(1, len(positions)):
        output_positions.append((positions[i] - positions[i - 1]) / value)
        output_time.append(time_stamps[i])

    return [output_time, output_positions]

def plot_graph(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

DISTANCES = []
VELOCITY = []
ACCELERATION = []
TIME_STAMPS = []

temp_time = 0
frame_id = 0

video_path = '../data/MVI_6591.MOV'  # Update this path accordingly
cap = cv2.VideoCapture(video_path)
scale = 0.018

while True:
    temp_time += 1 / 30
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break 
    TIME_STAMPS.append(temp_time)
    contours = contour_list(frame)
    if frame_id == 1:
        if N <= len(contours):
            initial_contour_id = N
            initial_point = contours[initial_contour_id - 1]
            values.append(0)
        else:
            print("Error")
    else:
        for i in contours:
            if euclidean_distance(i[1], initial_point[1]) < 10:
                initial_contour_id = i[0]
                values.append(euclidean_distance(i[1], initial_point[1]))
                break

DISTANCES.append(TIME_STAMPS)
DISTANCES.append(values)

VELOCITY = conversion(DISTANCES, 0.033)
ACCELERATION = conversion(VELOCITY, 0.033)

if len(VELOCITY[0]) == len(VELOCITY[1]) and len(ACCELERATION[0]) == len(ACCELERATION[1]):
    print("Sucess")
else:
    print("Failed")

# Plotting the graphs
m = min(len(DISTANCES[0]) , len(DISTANCES[1]))
DISTANCES[0] = DISTANCES[0][:m+1]
# DISTANCES[1] = DISTANCES[1][:m]
DISTANCES[1].append(DISTANCES[1][-1])
plot_graph(DISTANCES[0], DISTANCES[1], "Displacement vs Time", "Time (s)", "Displacement (mm)")
plot_graph(VELOCITY[0], VELOCITY[1], "Velocity vs Time", "Time (s)", "Velocity (mm/s)")
plot_graph(ACCELERATION[0], ACCELERATION[1], "Acceleration vs Time", "Time (s)", "Acceleration (mm/s^2)")

print(len(VELOCITY[0]), len(VELOCITY[1]))
print(len(ACCELERATION[0]), len(ACCELERATION[1]))
