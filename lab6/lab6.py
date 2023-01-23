import tello
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from disjoint_set import DisjointSet
import time

drone = tello.Tello('', 8887)


def main():
    
    time.sleep(10)

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    cv_file = cv2.FileStorage("out.xml", cv2.FILE_STORAGE_READ)
    intrinsic = cv_file.getNode("intrinsic").mat()
    distortion = cv_file.getNode("distortion").mat()
    cv_file.release()

    idx = 0
    
    print('start')
    drone.takeoff()

    while True:

        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        
        if (markerCorners):
            
            # Pose estimation for single markers
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            
            # Acquire the rotation matrix
            rotation_mat = np.array(cv2.Rodrigues(rvec)[0])

            # New z-axis after rotating
            new_z = np.array(rotation_mat.dot(z_axis))
            new_z[2] = -new_z[2]
            
            # Project the new z-axis to the xz-plane
            # v = np.array(new_z - new_z.dot(y_axis) * y_axis)
            # The rotating angle between the original z-axis and the new one
            # rad = math.atan2(v[0], v[2])

            rad = math.atan2(new_z[0], new_z[2])
            degree = math.degrees(rad)

            xDistance = tvec[0][0][0]
            yDistance = tvec[0][0][1]
            zDistance = tvec[0][0][2]
           
            print('new z: ', new_z)
            print('degree: ', degree)
            print('z_distance: ', tvec[0][0][2])
            print("y_distance: ", yDistance)

            if (degree > 15):
                drone.rotate_ccw(degree)
            elif (degree < -15):
                drone.rotate_cw(-degree)
            
            
            if (yDistance > 10):
                drone.move_down(yDistance/100)
            elif (yDistance < -10):
                drone.move_up(-yDistance / 100)

            if (xDistance > 10):
                drone.move_right(xDistance/100)
            elif (xDistance < -10):
                drone.move_left(-xDistance / 100)
            
            if (tvec[0][0][2] > 150):
                drone.move_forward(zDistance/200)
            
        # else:
        #     print('no!')
            

        cv2.imshow('output', frame)
        key = cv2.waitKey(1)
    

        if key != -1:
            drone.keyboard(key)
        
        


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()