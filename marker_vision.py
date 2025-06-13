import numpy as np
import cv2
import math

K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32)
D = np.zeros(5)  # [0, 0, 0, 0, 0]
cam = cv2.VideoCapture("http://192.168.137.221:4747/video")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

def get_aruco_pos(frame, drawing_frame = None):
    corners, ids, _ = aruco_det.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return

    # Display
    if drawing_frame is not None:
        cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
    
    # Return centroids and ids
    centroids = []
    for i, corner in enumerate(corners):
        corner = corner[0]  # Get the first (and only) element of the list
        centroid = np.mean(corner, axis=0)
        centroids.append(centroid)
    
    return np.array(centroids), ids.flatten()
    
if __name__ == "__main__":
    while True:
        # Get frame
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        drawing_frame = frame.copy()
        get_aruco_pos(frame, drawing_frame)

        cv2.imshow('Aruco Detection', drawing_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
            break
