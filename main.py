import cv2 
import numpy as np 
import pandas as pd


def euclidean_distance(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

def distances(lst):
    List = []
    for i in lst:
        List.append(euclidean_distance(i[0][1], i[1][1]))
    return List

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
            cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)
            cv2.putText(frame, f"{contour_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frame_points.append([contour_name,[cX, cY]])
    # cv2.imshow("FRAME", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return frame_points

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
    cv2.imshow("FRAME", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def highlight_points(frame,coordinates,text):
    tpl = (coordinates[0] , coordinates[1])
    cv2.circle(frame,tpl, 1, (0, 0, 255), -1)
    cv2.putText(frame, f"{text}",tpl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

video_path = 'C:/Users/SURYA/Desktop/Surya/code_raja_code/ComputerVisionProject_MohanSC/Data/MVI_6591.mp4'
cap = cv2.VideoCapture(video_path)

frame_id = 0

pairs = []

interested_points = []

DISTANCES = []

DISTANCES_NORMALIZED = []

LINE_WISE_NORMALIZED_DISTANCES = []

LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES = []


TIME_STAMPS = []
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
        spl_frame = frame.copy()
        initial_list_of_points = contour_list(frame)
        print("Enter the number of lines. ")
        lines = int(input())
        for i in range(lines):
            print("Enter the points which make Line ", i+1)
            user_input = input()
            points = user_input.split()
            interested_points.append([initial_list_of_points[int(points[0])-1],initial_list_of_points[int(points[1])-1]])
            # print(initial_list_of_points[int(points[0])-1])
            # print(type(initial_list_of_points[int(points[0])]))
            spl_frame = highlight_points(spl_frame,initial_list_of_points[int(points[0])][1],points[0])
            spl_frame = highlight_points(spl_frame,initial_list_of_points[int(points[1])][1],points[1])
    else:
        frame_points = contour_list(frame)
        for i in frame_points:
            for j in interested_points:
                if euclidean_distance(i[1],j[0][1]) < 5:
                    j[0][0] = i[0]
                    j[0][1] = i[1]
                if euclidean_distance(i[1],j[1][1]) < 5:
                    j[1][0] = i[0]
                    j[1][1] = i[1]
        for i in interested_points:
            frame1 = highlight_points(frame1,i[0][1],i[0][0])
            frame1 = highlight_points(frame1,i[1][1],i[1][0])
        cv2.imshow("FRAME", frame1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # print(distances(interested_points))
    DISTANCES.append(distances(interested_points))
# print(DISTANCES)
print("````````````````````````````````````````````")

scale = 4/DISTANCES[0][0]
print(scale)

for i in range(len(DISTANCES[0])):
    temp = []
    for j in range(len(DISTANCES)):
        temp.append(DISTANCES[j][i]*scale)
    LINE_WISE_NORMALIZED_DISTANCES.append(temp)
# print(LINE_WISE_NORMALIZED_DISTANCES)

# print(len(LINE_WISE_NORMALIZED_DISTANCES))


# print(TIME_STAMPS)

# print("****************")

# print(len(TIME_STAMPS))

for i in LINE_WISE_NORMALIZED_DISTANCES:
    temp = []
    for j in i:
        temp.append(4-j)
    LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES.append(temp)
    


print(len(LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES))

print(len(LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES[0]))

AVG = []

for i in range(len(LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES[0])):
    n = 0
    avg = 0
    for j in range(len(LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES)):
        avg += LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES[j][i]
        n += 1
    avg = avg/n
    print(avg)
    AVG.append(avg)
print(len(AVG))
LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES.append(AVG)

LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES.insert(0,TIME_STAMPS)

print(len(LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES))

df = pd.DataFrame(LINE_WISE_ERRORS_IN_NORMALIZED_DISTANCES).transpose()

# Specify the path and file name for the Excel file
excel_file_path = "output.xlsx"

# Write the DataFrame to the Excel file
df.to_excel(excel_file_path, index=False, header=False)  # index and header are set to False

print(f"Data written to {excel_file_path} successfully.")