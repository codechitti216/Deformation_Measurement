import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Video Tracking Application")

    # Step 1: Video Uploading
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Read the video using OpenCV
        video = cv2.VideoCapture(uploaded_file.name)
        
        # Step 2: Showing the first frame of the video
        ret, first_frame = video.read()
        if ret:
            # Convert the frame from BGR to RGB
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.image(first_frame_rgb, caption="First Frame of Video", use_column_width=True)
            
            # Step 3: Drawing Contours on the first frame and plotting centroids
            gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroids = []
            contour_id = 1
            contour_frame = first_frame.copy()
            for contour in contours:
                contour_id += 1
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append([cx, cy])
                    cv2.putText(contour_frame, f"{contour_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(contour_frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Display the frame with contours and centroids
            contour_frame_rgb = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2RGB)
            st.image(contour_frame_rgb, caption="Contours and Centroids on First Frame", use_column_width=True)
            
            # Step 4: Asking the user which centroids to track
            centroid_options = [f"Centroid {i + 1}: ({cx}, {cy})" for i, (cx, cy) in enumerate(centroids)]
            selected_centroid_indices = st.multiselect("Select centroids to track",centroid_options)
            print(selected_centroid_indices)
            # Convert selected centroid indices to list of tuples (coordinates)
            selected_centroids = [centroids[int(idx.split(":")[0].split(" ")[1]) - 1] for idx in selected_centroid_indices]
            
            if selected_centroids:
                # Create a black image for plotting trajectories
                height, width, _ = first_frame.shape
                trajectory_image = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Initialize colors for each selected centroid
                colors = [tuple(np.random.randint(0, 255, size=3)) for _ in selected_centroids]
                
                # Track each selected centroid on subsequent frames
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    for idx, (cx, cy) in enumerate(selected_centroids):
                        # Tracking each point using mean-shift
                        roi = cv2.circle(trajectory_image, (cx, cy), 5, colors[idx], -1)
                    
                    # Show the trajectory image
                    trajectory_rgb = cv2.cvtColor(trajectory_image, cv2.COLOR_BGR2RGB)
                    st.image(trajectory_rgb, caption="Trajectory of Selected Centroids", use_column_width=True)
                
            video.release()

if __name__ == "__main__":
    main()