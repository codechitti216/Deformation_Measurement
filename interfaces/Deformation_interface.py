import streamlit as st
import cv2
import numpy as np
import os
import time
import pandas as pd

def process_video(video_path, contour_id1, contour_id2, scale, original_distance):
    
    interested_points = []
    
    DISTANCES = []
    
    LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES = []
    
    TIME_STAMPS = []
    
    def distances(lst):
        List = []
        List.append(euclidean_distance(lst[0][1], lst[1][1]))
        return List
        
    def return_image(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, binary = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        

        contour_name = 0
        for idx, contour in enumerate(contours):
            M = cv2.moments(contour)
            contour_name += 1
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)
                cv2.putText(frame, f"{contour_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame
    
    def draw_line_and_distance(frame, point1, point2):
        # Draw line between the points on the frame
        cv2.line(frame, point1, point2, (0, 255, 0), 2)
        
        # Calculate Euclidean distance between the points
        distance = euclidean_distance(point1, point2)
        
        # Put the distance value beside the line
        cv2.putText(frame, f'{distance:.2f}', ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show the frame with the line and distance
    
    def highlight_points(frame,coordinates,text):
        tpl = (coordinates[0] , coordinates[1])
        cv2.circle(frame,tpl, 1, (0, 0, 255), -1)
        cv2.putText(frame, f"{text}",tpl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame
    
    def Show_image(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, binary = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        

        contour_name = 0
        for idx, contour in enumerate(contours):
            M = cv2.moments(contour)
            contour_name += 1
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)
                cv2.putText(frame, f"{contour_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    
    def euclidean_distance(point1, point2):
        return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

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

    cap = cv2.VideoCapture(video_path)

    
    frame_id = 0
    
    temp_time = 0
    
    while True:
        temp_time += 1/30
        TIME_STAMPS.append(temp_time)
        frame_id += 1
        ret, frame = cap.read()
        if not ret:
            break 
        frame2 = frame.copy()
        frame1 = frame.copy()
        if frame_id == 1:
            Show_image(frame2)
            cv2.waitKey(0)
            cv2.destroyAllWindows
            spl_frame = frame.copy()
            initial_list_of_points = contour_list(frame)
            interested_points.append([initial_list_of_points[int(contour_id1[0])-1],initial_list_of_points[int(contour_id2[0])-1]])
            spl_frame = highlight_points(spl_frame,initial_list_of_points[int(contour_id1[0])][1],contour_id1[0])
            spl_frame = highlight_points(spl_frame,initial_list_of_points[int(contour_id2[0])][1],contour_id2[0])
        else:
            frame1 = return_image(frame1)
            frame_points = contour_list(frame)
            for i in frame_points:
                for j in interested_points:
                    if euclidean_distance(i[1],j[0][1]) < 10:
                        j[0][0] = i[0]
                        j[0][1] = i[1]
                    if euclidean_distance(i[1],j[1][1]) < 10:
                        j[1][0] = i[0]
                        j[1][1] = i[1]
            for i in interested_points:
                frame1 = highlight_points(frame1,i[0][1],i[0][0])
                frame1 = highlight_points(frame1,i[1][1],i[1][0])
                draw_line_and_distance(frame1,i[0][1],i[1][1])
        DISTANCES.append(distances(interested_points[0])[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows
    for i in DISTANCES:
        LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES.append(original_distance - (i*scale))
        
       
    return LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES

def euclidean_distance(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

def contour_list(frame):
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
            frame_points.append([contour_name,[cX, cY]])

    return frame_points

def highlight_points(frame,coordinates,text):
    tpl = (coordinates[0] , coordinates[1])
    cv2.circle(frame,tpl, 1, (0, 0, 255), -1)
    cv2.putText(frame, f"{text}",tpl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


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

def plain_frames_extract(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def save_uploaded_file(uploaded_file):
    try:
        # Define a directory to save the uploaded file
        save_dir = "uploads"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Create the file path
        file_path = os.path.join(save_dir, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def calculate_scale(points,actual_length):
    scale = actual_length/euclidean_distance(points[0][1],points[1][1])
    return scale

def main():
    st.title("Interface 1.0")
    
    points_Window_1 = []

    video_file = st.file_uploader("Upload a video", type=["mp4","MTS "])

    if video_file:
        video_path = save_uploaded_file(video_file)
        frame_index = 0
        frames = extract_frames(video_path)
        plain_frames = plain_frames_extract(video_path)
        frame_show = plain_frames[0]
        if "frame_index" not in st.session_state:
                st.session_state.frame_index = 0

        image_location = st.empty()
        image_location.image(frames[st.session_state.frame_index], channels="RGB", use_column_width=True)

        col1, col2, col3, col4 = st.columns(4)
        
        # Use a single image location for displaying the fram

        if col2.button("Previous Frame") and st.session_state.frame_index > 0:
            st.session_state.frame_index -= 1
            image_location.image(frames[st.session_state.frame_index], channels="RGB", use_column_width=True)

        if col3.button("Next Frame") and st.session_state.frame_index < len(frames) - 1:
            st.session_state.frame_index += 1
            image_location.image(frames[st.session_state.frame_index], channels="RGB", use_column_width=True)

        if col4.button("Fast Forward"):
            for i in range(st.session_state.frame_index, len(frames)):
                image_location.image(frames[i], channels="RGB", use_column_width=True)
                time.sleep(0.1)  # Adjust sleep duration to control playback speed
                st.session_state.frame_index = i


        options= contour_list(plain_frames[0])
        num_lines = st.number_input("Enter the number of lines", min_value=1, value=1, step=1)

        scales = []
        for i in range(num_lines):
            st.write(f"Line {i + 1}")
            point1 = st.selectbox(f"Select the contour id of the first point of the line {i+1}", options, key=f"contour_id_{i}_point1")
            frame_show = highlight_points(plain_frames[0],point1[1],str(point1[0]))
            point2 = st.selectbox(f"Select the contour id of the second point of the line {i+1}", options, key=f"contour_id_{i}_point2")
            frame_show = highlight_points(frame_show,point2[1],str(point2[0]))
            image_location = st.empty()
            image_location.image(frame_show, channels="RGB", use_column_width=True)
                
                                
            actual_length = st.number_input("Enter actual length for the line", value=0.0, step=0.01, key=f"actual_length_{i}")
            if actual_length is not None: 
                try:
                    scale = calculate_scale([point1,point2], actual_length)
                    points_Window_1.append([scale, point1, point2, actual_length])
                    scales.append(scale)
                except ZeroDivisionError:
                    st.error("Actual length cannot be 0. Please enter a non-zero value.")

        if scales:
            average_scale = sum(scales) / len(scales)
            st.write(f"Average Scale: {average_scale}")
    
    if st.button("Old points"):
        k = 0
        output = []
        for i in range(num_lines):
            k += 1
            temp = []
            # st.write("This is the defitmatipon between ",points_Window_1[i][1][0]," and ",points_Window_1[i][2][0])
            temp = process_video(video_path,points_Window_1[i][1],points_Window_1[i][2],points_Window_1[i][0], points_Window_1[i][3])
            temp = ["Deformation of the line " + str(k)+ "   "] + temp
            output.append(temp)
        df = pd.DataFrame(output)
        excel_path = "output.xlsx"
        df.to_excel(excel_path, index=False)
        st.write("Excel sheet generated. Here is the content:")
        st.write(df)  # Display the DataFrame
        

    if st.button("New Pairs"):
        st.write("New pairs")
        # frame_index = 0
        # frames = extract_frames(video_path)
        # plain_frames = plain_frames_extract(video_path)
        # frame_show = plain_frames[0]
        # image_location1 = st.empty()
        # image_location1.image(frames[frame_index], channels="RGB", use_column_width=True)

        # options= contour_list(plain_frames[0])
        # num_lines = st.number_input("Enter the number of lines", min_value=0, value=1, step=1)

        # scales = []
        # for i in range(num_lines):
        #     st.write(f"Line {i + 1}")
        #     point1 = st.selectbox(f"Select contour ID for line {i+1}", options, key=f"contour_id_{i}_point3")
        #     frame_show = highlight_points(plain_frames[0],point1[1],str(point1[0]))
        #     point2 = st.selectbox(f"Select contour ID for line {i+1}", options, key=f"contour_id_{i}_point4")
        #     frame_show = highlight_points(frame_show,point2[1],str(point2[0]))
        #     image_location = st.empty()
        #     image_location.image(frame_show, channels="RGB", use_column_width=True)
            
                        
        #     actual_length = st.number_input("Enter actual length for the line", value=0.0, step=0.01, key=f"actual_leng_{i}")
        #     if actual_length is not None: 
        #         try:
        #             scale = calculate_scale([point1,point2], actual_length)
        #             points_Window_1.append([scale, point1, point2, actual_length])
        #             scales.append(scale)
        #         except ZeroDivisionError:
        #             st.error("Actual length cannot be 0. Please enter a non-zero value.")
                    
        #     for i in range(num_lines):
        #         st.write("This is the defitmatipon between ",points_Window_1[i][1][0]," and ",points_Window_1[i][2][0])
        #         output = process_video(video_path,points_Window_1[i][1],points_Window_1[i][2],points_Window_1[i][0], points_Window_1[i][3])
        #         st.write(output)


        # if scales:
        #     average_scale = sum(scales) / len(scales)
        #     st.write(f"Average Scale: {average_scale}")
        
    if st.button("Exit"):
            st.stop()
            
            
if __name__ == "__main__":
    main()
