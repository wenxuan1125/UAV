import tello
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import time

act_state = 1 # 0->L,1->R,2->U,3->D

drone = tello.Tello('', 8080)
aruco4_counter=0


def get_dir(img):
    hight, width = img.shape
    rec = list()
    leftmin, leftmax, rightmin, rightmax = None,None,None,None
    for h in range(hight):
        if(img[h,0] == 255):
            if leftmin == None:
                leftmin = h
            leftmax = max(h, leftmax) if leftmax is not None else h
        if(img[h,width-1] == 255):
            if rightmin == None:
                rightmin = h
            rightmax = max(h, rightmax) if rightmax is not None else h
    
    rec.append((leftmin, leftmax))
    rec.append((rightmin,rightmax))

    topmin, topmax, bottommin, bottommax = None,None,None,None
    for w in range(width):
        if(img[0,w] == 255):
            if topmin == None:
                topmin = w
            topmax = max(w, topmax) if topmax is not None else w
        if(img[hight-1, w] == 255):
            if bottommin == None:
                bottommin = w
            bottommax = max(w, bottommax) if bottommax is not None else w
    
    rec.append((topmin, topmax))
    rec.append((bottommin,bottommax))

    return rec # true->上下, False->左右

def main():
    
    global aruco4_counter, act_state
    cap = cv2.VideoCapture(1)
    time.sleep(10)

    state = list()

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    cv_file = cv2.FileStorage("out (2).xml", cv2.FILE_STORAGE_READ)
    intrinsic = cv_file.getNode("intrinsic").mat()
    distortion = cv_file.getNode("distortion").mat()
    cv_file.release()
    
    print(drone.get_battery())
    
    # make drone take off
    drone.takeoff()
    time.sleep(8)

    begin_time = time.time()
    key = -1
    delta = 50
    while True:

        # frame = drone.read()
        ret, frame = cap.read()
        hight, width, channel = frame.shape
        current_time = time.time()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        cv2.imshow('output', frame)
        key = cv2.waitKey(1)

        if (markerCorners):
            # Pose estimation for single markers
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

            xDistance = tvec[0][0][0]
            yDistance = tvec[0][0][1]
            zDistance = tvec[0][0][2]

            if(zDistance>300):
                continue
            
            if (markerIds == 4 ):
                if(current_time - begin_time > 60):
                    aruco4_counter-=2
                    drone.land()
                    

                ### make sure it in the center of y-axis
                if (yDistance > 10):
                    drone.move_up(0.2)
                    key = cv2.waitKey(3000)
                    drone.move_down(0.2+yDistance/100)
                    key = cv2.waitKey(3000)

                elif (yDistance < -10):
                    drone.move_down(0.2)
                    key = cv2.waitKey(3000)
                    drone.move_up(0.2 - yDistance/100)
                    key = cv2.waitKey(3000)
                
                if(aruco4_counter==0):
                    aruco4_counter+=1
                    drone.move_right(0.3)
                    cv2.waitKey(3000)

                elif(aruco4_counter==1):
                    drone.move_right(0.2)
                    cv2.waitKey(3000)

        
        # no marker corner, trace the blue line
        elif(aruco4_counter==1):
            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # define range of color in HSV
            #blue
            lower = np.array([100,50,50])
            upper = np.array([130,255,255])

            # Threshold the HSV image to get only colors
            mask = cv2.inRange(hsv, lower, upper)

            rec = get_dir(mask)

            #cv2.imshow('frame',frame)
            # cv2.imshow('mask',mask)
            #cv2.waitKey(33)

            # print(rec)
            if None not in rec[act_state]:
                if(act_state==0):
                    drone.move_left(0.2)
                    # print(act_state)
                    cv2.waitKey(3000)
                if(act_state==1):
                    drone.move_right(0.2)
                    # print(act_state)
                    cv2.waitKey(3000)
                if(act_state==2):
                    drone.move_up(0.2)
                    # print(act_state)
                    cv2.waitKey(3000)
                if(act_state==3):
                    drone.move_down(0.2)
                    # print(act_state)
                    cv2.waitKey(3000)

            elif(act_state<=1):
                if None not in rec[2]:
                    drone.move_up(0.2)
                    act_state = 2
                    # print(act_state)
                    cv2.waitKey(3000)
                else:
                    drone.move_down(0.2)
                    act_state = 3
                    # print(act_state)
                    cv2.waitKey(3000)

            elif(act_state>=2):
                if None not in rec[0]:
                    drone.move_left(0.2)
                    act_state = 0
                    # print(act_state)
                    cv2.waitKey(3000)
                else:
                    drone.move_right(0.2)
                    act_state = 1
                    # print(act_state)
                    cv2.waitKey(3000)
            
    

        if key != -1:
            drone.keyboard(key)
            cv2.waitKey(5000)
        
        


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()