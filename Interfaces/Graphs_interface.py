import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os

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
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def process_video(video_path, N, scale):
    values = []
    DISTANCES = []
    VELOCITY = []
    ACCELERATION = []
    TIME_STAMPS = []

    temp_time = 0
    frame_id = 0

    cap = cv2.VideoCapture(video_path)

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
                st.error("Error: N exceeds the number of contours.")
                return None, None, None
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
    
    if len(DISTANCES[0]) != len(DISTANCES[1]):
        m = min(len(DISTANCES[0]) , len(DISTANCES[1]))
        DISTANCES[0] = DISTANCES[0][:m+1]
        DISTANCES[1].append(DISTANCES[1][-1])
    return DISTANCES, VELOCITY, ACCELERATION

def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        file_path = os.path.join("../data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def show_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, binary = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)
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
            frame_points.append([contour_name,[cX, cY]])
    return frame

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(show_image(frame_rgb))
    cap.release()
    return frames

def main():
    st.title("Video Contour Tracking and Graph Plotting")

    video_file = st.file_uploader("Upload a video", type=["mp4","MTS"])
    
    video_path = save_uploaded_file(video_file)
    frame_index = 0
    frames = extract_frames(video_path)
    if frames:
        image_location1 = st.empty()
        image_location1.image(frames[frame_index], channels="RGB", use_column_width=True)

    if video_file:
        video_path = save_uploaded_file(video_file)
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        N = st.number_input("Enter contour ID to track", min_value=1, value=1, step=1)
        scale = 0.018

        if st.button("Generate Graphs"):
            DISTANCES, VELOCITY, ACCELERATION = process_video(video_path, N, scale)

            if DISTANCES is not None:
                st.title("Graphs for the contours ")
                dist_graph = plot_graph(DISTANCES[0], DISTANCES[1], "Displacement vs Time", "Time (s)", "Displacement (mm)")
                vel_graph = plot_graph(VELOCITY[0], VELOCITY[1], "Velocity vs Time", "Time (s)", "Velocity (mm/s)")
                acc_graph = plot_graph(ACCELERATION[0], ACCELERATION[1], "Acceleration vs Time", "Time (s)", "Acceleration (mm/s^2)")

                st.image(dist_graph, caption="")
                st.image(vel_graph, caption="")
                st.image(acc_graph, caption="")

if __name__ == "__main__":
    main()