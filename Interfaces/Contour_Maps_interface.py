import streamlit as st
import cv2
import numpy as np
from scipy.interpolate import griddata

def Contour_Creator(video_path):

    def euclidean_distance(point1, point2):
        return np.sqrt(((point1[0] - point2[0])**2) + ((point1[1] - point2[1]) ** 2))

    def distance(point1, point2):
        return [abs((int(point1[0]) - int(point2[0]))) , abs((int(point1[1]) - int(point2[1])))]

    def contour_list(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                frame_points.append([contour_name, [cX, cY]])
        return frame_points

    def value_generator(i, point_list):
        x = 0
        y = 0
        distances = []
        for j in point_list:
            distances.append([euclidean_distance(i[1], j[1]), j[1]])
        distances.sort()
        for k in range(4):
            x += distance(i[1], distances[k][1])[0]
            y += distance(i[1], distances[k][1])[1]
        return [i[1], x/4], [i[1], y/4]

    def create_heatmap(image, data_points):
        # Extract x, y coordinates and values from the data points
        x = np.array([point[0][0] for point in data_points])
        y = np.array([point[0][1] for point in data_points])
        values = np.array([point[1] for point in data_points])

        # Create a grid to interpolate values
        xi, yi = np.linspace(0, image.shape[1], image.shape[1]), np.linspace(0, image.shape[0], image.shape[0])
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the values to the grid
        zi = griddata((x, y), values, (xi, yi), method='linear')

        # Normalize the interpolated values to the range 0-255
        zi_normalized = cv2.normalize(zi, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8
        zi_normalized = np.uint8(zi_normalized)

        # Apply a color map to create the heat map
        heatmap = cv2.applyColorMap(zi_normalized, cv2.COLORMAP_JET)

        # Overlay the heat map on the original image
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

        return overlay

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    
    framesx = []
    framesy = []
    while True:
        valuesx = []
        valuesy = []
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break 
        frame_id += 1
        initial_points = contour_list(frame)
        
        for i in initial_points:
            x , y = value_generator(i, initial_points)
            valuesx.append(x)
            valuesy.append(y)
        
        heatmap_imageX, heatmap_imageY = create_heatmap(frame, valuesx), create_heatmap(frame,valuesy)

        # Overlay heatmap onto original frame
        framesx.append(cv2.addWeighted(frame, 0.5, heatmap_imageX, 0.5, 0))
        framesy.append(cv2.addWeighted(frame, 0.5, heatmap_imageY, 0.5, 0))
    return framesx, framesy

# Streamlit app
st.title('Contour Creator Heatmap Generator')

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Generate heatmaps
    framesx, framesy = Contour_Creator("temp_video.mp4")

    # Display the heatmaps for the first frame as a sample
    st.write("Sample heatmaps for the first frame:")

    st.image(framesx[0], caption="Heatmap X Direction", use_column_width=True)
    st.image(framesy[0], caption="Heatmap Y Direction", use_column_width=True)

# Optionally, provide a way to download the heatmap frames
# For simplicity, this example does not include the download implementation
