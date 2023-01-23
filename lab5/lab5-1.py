import tello
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from disjoint_set import DisjointSet
import time

drone = tello.Tello('', 8887)


def camera_calibration():

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    row = 6
    column = 9
    frame_num = 8
    i = 0

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row*column,3), np.float32)
    objp[:,:2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # calibration
    while(True):
        frame = drone.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret2, corners = cv2.findChessboardCorners(gray_frame, (row, column), None)

        # If found, add object points, image points (after refining them)
        if ret2 == True:
        
            # Optmized the found corners
            optimized_corners = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(optimized_corners)

            i = i + 1
            if (i == frame_num):
                break

            time.sleep(4)


        cv2.imshow('frame', frame)
        cv2.waitKey(33)

 
    # calculate the parameters
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

    # store the parameters
    f = cv2.FileStorage('out.xml', cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()

    print('calibration completed!')

def marker_detection():

    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    cv_file = cv2.FileStorage("out.xml", cv2.FILE_STORAGE_READ)
    intrinsic = cv_file.getNode("intrinsic").mat()
    distortion = cv_file.getNode("distortion").mat()
    cv_file.release()

    print("read matrix\n", intrinsic)
    print('here')
    print("read matrix\n", distortion)

    idx = 0 

    while True:

        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        
        if (markerCorners):
            
            # Pose estimation for single markers
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15 , intrinsic, distortion)

            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 10)

            #cv2.putText( img, text to show, coordinate, font, font size, color, line width, line type )
            cv2.putText(frame,'x : '+ str(tvec[0][0][0]), (40, 20), cv2.FONT_HERSHEY_SIMPLEX,\
                0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame,'y : ' +str(tvec[0][0][1]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX,\
                0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame,'z : '+str(tvec[0][0][2]), (40, 60), cv2.FONT_HERSHEY_SIMPLEX,\
                0.4, (0, 255, 255), 1, cv2.LINE_AA)
            
        cv2.imshow('output', frame)
        cv2.waitKey(33)


    


def main():
    
    time.sleep(10)

    camera_calibration()
    marker_detection()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()