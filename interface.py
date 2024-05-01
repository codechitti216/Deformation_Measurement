import streamlit as st
import cv2
import numpy as np
import os
import time

def euclidean_distance(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

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
    # cv2.imshow("FRAME", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

def save_uploaded_file(uploaded_file):
    file_path = os.path.join("C:/Users/SURYA/Desktop/Surya/code_raja_code/ComputerVisionProject_MohanSC/Data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def calculate_scale(points,actual_length):
    scale = actual_length/euclidean_distance(points[0],points[1])
    return scale

def main():
    st.title("Video Viewer and Scale Estimation")

    video_file = st.file_uploader("Upload a video", type=["mp4"])

    if video_file:
        video_path = save_uploaded_file(video_file)
        frames = extract_frames(video_path)
        frame_index = 0

        image_location = st.empty()
        image_location.image(frames[frame_index], channels="RGB", use_column_width=True)

        col1, col2, col3, col4 = st.columns(4)

        if col2.button("Previous Frame") and frame_index > 0:
            frame_index -= 1
            image_location.image(frames[frame_index], channels="RGB", use_column_width=True)

        if col3.button("Next Frame") and frame_index < len(frames) - 1:
            frame_index += 1
            image_location.image(frames[frame_index], channels="RGB", use_column_width=True)

        if col4.button("Fast Forward"):
            for i in range(frame_index, len(frames)):
                image_location.image(frames[i], channels="RGB", use_column_width=True)
                time.sleep(0.1)  # Adjust sleep duration to control playback speed

        if st.button("Exit"):
            st.stop()

        
        num_lines = st.number_input("Enter the number of lines", min_value=1, value=1, step=1)

        scales = []
        for i in range(num_lines):
            st.write(f"Line {i + 1}")
            points = []
            for j in range(2):  # Assuming each line requires 2 points
                point_input = st.text_input(f"Enter coordinates for point {j+1} (format: x,y)", key=f"point_{i}_{j}")
                if point_input.strip():  # Check if input string is not empty
                    try:
                        
                        point = tuple(map(int, point_input.split(',')))
                        points.append(point)
                    except ValueError:
                        st.error("Invalid input format. Please enter coordinates in the format: x,y")
            actual_length = st.number_input("Enter actual length for the line", value=0.0, step=0.01, key=f"actual_length_{i}")
            if actual_length is not None: 
                try:
                    scale = calculate_scale(points, actual_length)
                    scales.append(scale)
                except ZeroDivisionError:
                    st.error("Actual length cannot be 0. Please enter a non-zero value.")


        if scales:
            average_scale = sum(scales) / len(scales)
            st.write(f"Average Scale: {average_scale}")
if __name__ == "__main__":
    main()
