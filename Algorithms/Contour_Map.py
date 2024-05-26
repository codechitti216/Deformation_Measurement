import cv2
import numpy as np
from scipy.interpolate import griddata

# Global variables to store selected points
point1 = None
point2 = None
selected_points = []

def Contour_Creator(video_path):

    def euclidean_distance(point1, point2):
        return np.sqrt(((point1[0] - point2[0])**2) + ((point1[1] - point2[1]) ** 2))

    def distance(point1, point2):
        return [abs((int(point1[0]) - int(point2[0]))) , abs((int(point1[1]) - int(point2[1])))]

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
                frame_points.append([contour_name, [cX, cY]])
        return frame_points

    def value_generator(i, point_list):
        x = 0
        y = 0
        distances = []
        for j in point_list:
            distances.append([euclidean_distance(i[1], j[1]), j[1]])
        distances.sort(key=lambda x: x[0])
        for k in range(min(4, len(distances))):
            x += distance(i[1], distances[k][1])[0]
            y += distance(i[1], distances[k][1])[1]
        return [i[1], x / 4], [i[1], y / 4]

    def create_heatmap(image, data_points):
        x = np.array([point[0][0] for point in data_points])
        y = np.array([point[0][1] for point in data_points])
        values = np.array([point[1] for point in data_points])

        xi, yi = np.linspace(0, image.shape[1], image.shape[1]), np.linspace(0, image.shape[0], image.shape[0])
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), values, (xi, yi), method='linear')

        zi_normalized = cv2.normalize(zi, None, 0, 255, cv2.NORM_MINMAX)

        zi_normalized = np.uint8(zi_normalized)

        heatmap = cv2.applyColorMap(zi_normalized, cv2.COLORMAP_JET)

        return heatmap

    def filter_points_by_range(points, x_range, y_range):
        filtered_points = []
        for point in points:
            if x_range[0] <= point[1][0] <= x_range[1] and y_range[0] <= point[1][1] <= y_range[1]:
                filtered_points.append(point)
        return filtered_points

    def mouse_event_handler(event, x, y, flags, param):
        global point1, point2, selected_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if point1 is None:
                point1 = (x, y)
            else:
                point2 = (x, y)
                selected_points = [(min(point1[0], point2[0]), min(point1[1], point2[1])),
                                   (max(point1[0], point2[0]), max(point1[1], point2[1]))]

    cap = cv2.VideoCapture(video_path)
    frames = []

    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", mouse_event_handler)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        # Clone the frame to avoid modifying the original
        frame_copy = frame.copy()

        if len(selected_points) == 2:
            initial_points = contour_list(frame_copy)

            # Filter points by selected region
            filtered_points = filter_points_by_range(initial_points, selected_points[0], selected_points[1])

            values = []
            for point in filtered_points:
                x, y = value_generator(point, filtered_points)
                values.append((x, y))

            # Create heatmap for selected region
            heatmap_image = create_heatmap(frame_copy, values)

            # Overlay heatmap onto original frame
            result = cv2.addWeighted(frame_copy, 0.5, heatmap_image, 0.5, 0)

            # Reset selected points
            point1 = None
            point2 = None
            selected_points = []

            frames.append(result)

            cv2.imshow("Heatmap for Selected Region", result)

        else:
            cv2.imshow("Select Region", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames

# Example usage
Contour_Creator("C:/Users/SURYA/Desktop/Surya/code_raja_code/ComputerVisionProject_MohanSC/Data/data/MVI_6591.mp4")
