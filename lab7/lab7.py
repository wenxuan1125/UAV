import tello
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from disjoint_set import DisjointSet
import time

drone = tello.Tello('', 8080)


def main():
    
    time.sleep(10)

    state = list()

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
    
    print(drone.get_battery())
    
    ### make drone take off
    # drone.takeoff()
    # time.sleep(5)

    pret = time.time()
    key = -1
    a =1
    while True:

        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        cv2.imshow('output', frame)
        key = cv2.waitKey(1)

        if (markerCorners):
            # Pose estimation for single markers
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            if (a == 1):
                print(rvec)
                a+=1
            if(markerIds.size>1):
                rvec = rvec[0][0]
                markerIds = markerIds[0]
                print('here',rvec,markerIds)
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
           
            # print('new z: ', new_z)
            # print('degree: ', degree)
            # print('z_distance: ', tvec[0][0][2])
            # print("y_distance: ", yDistance)

            if(zDistance>300):
                continue

            
            if (markerIds == 0 ):
                # no command
                if (len(state) == 0):
                    pret = time.time()
                    # # move backward
                    # if (zDistance < 50):
                    #     drone.move_backward(0.2)
                    #     state.append('mb')
                    # move forward
                    if (zDistance > 100):
                        drone.move_forward(0.4)
                        state.append('mf')

                    # rotate
                    elif (degree > 15):
                        drone.rotate_ccw(20)
                        state.append('rccw')
                    elif (degree < -15):
                        drone.rotate_cw(20)
                        state.append('rcw')
                    
                    # up and down
                    elif (yDistance > 10):
                        drone.move_down(0.2)
                        state.append('md')
                    elif (yDistance < -10):
                        drone.move_up(0.2)
                        state.append('mu')

                    # left and right
                    elif (xDistance > 10):
                        drone.move_right(0.2)
                        state.append('mr')
                    elif (xDistance < -10):
                        drone.move_left(0.2)
                        state.append('ml')

                    

                # if the command has been executed, clear the state list so that the following command can be assigned
                else:
                    if state == 'rccw' and degree <= 15:
                        state = []
                    elif state == 'rcw' and degree >= -15:
                        state = []
                    elif state == 'md' and yDistance <= 10:
                        state = []
                    elif state == 'mu' and yDistance >= -10:
                        state = []
                    elif state == 'mr' and xDistance <= 10:
                        state = []
                    elif state == 'ml' and xDistance >= -10:
                        state = []
                    elif state == 'mf' and zDistance <= 90:
                        state = []
                    elif state == 'mb' and zDistance >= 50:
                        state = []

                    else:
                        state = []

                    if((time.time() - pret)>3):
                        state = []

            elif (markerIds == 1 ):
                drone.rotate_ccw(180)
                key = cv2.waitKey(6000)

            elif (markerIds == 2 ):
                if (xDistance > 0):
                    drone.move_right(1)
                    key = cv2.waitKey(3000)
                
                if key != -1:
                    drone.keyboard(key)
                    cv2.waitKey(5000)

            elif (markerIds == 3 ):

                # # move forward
                if (zDistance > 100):
                    '''
                    先做校正
                    '''
                    # up and down
                    if (yDistance > 15):
                        drone.move_down(0.2)
                        key = cv2.waitKey(3000)
                    elif (yDistance < -15):
                        drone.move_up(0.2)
                        key = cv2.waitKey(3000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)

                    # left and right
                    if (xDistance > 20):
                        drone.move_right(xDistance/100)
                        key = cv2.waitKey(3000)
                    elif (xDistance < -15):
                        drone.move_left(-xDistance/100)
                        key = cv2.waitKey(3000)
                    
                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)
                    
                    # after moving, the distance between the drone and aruco is 70 cm
                    drone.move_forward(0.4)
                    key = cv2.waitKey(5000)

                    continue
                
                drone.move_forward(0.2)  
                cv2.waitKey(5000)
                

                # move down
                res = 'none_response'
                while(('dm' not in res) and (res != 'ok') ):
                    res = drone.move_down(0.6)
                    cv2.waitKey(5000)
                    print(res)
                print('output',res)
                
                # move forward to cross the board
                res = 'none_response'
                while(res != 'ok'):
                    res = drone.move_forward((zDistance+70)/100)
                    print(res)
                    key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)

                print('output',res)

                # move up
                res = 'none_response'
                while(res != 'ok'):
                    res = drone.move_up(1)
                    cv2.waitKey(5000)

                print(res)

            elif (markerIds == 4 ):

                # move forward
                if (zDistance > 100):
                    '''
                    先做校正
                    '''
                    # up and down
                    if (yDistance > 15):
                        drone.move_down(0.2)
                        key = cv2.waitKey(3000)
                    elif (yDistance < -15):
                        drone.move_up(0.2)
                        key = cv2.waitKey(3000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)

                    # left and right
                    if (xDistance > 20):
                        drone.move_right(xDistance/100)
                        key = cv2.waitKey(3000)
                    elif (xDistance < -15):
                        drone.move_left(-xDistance/100)
                        key = cv2.waitKey(3000)
                    
                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)
                    # after moving, the distance between the drone and aruco is 70 cm
                    drone.move_forward(0.4)
                    
                    key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)
                    continue

                

                # move up
                res = 'none_response'
                while(('dm' not in res) and (res != 'ok') ):
                    res = drone.move_up(0.8)
                    print(res)
                    key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)

                print('output',res)
                time.sleep(5)
                
                # move forward to cross the board
                res = 'none_response'
                while(res != 'ok'):
                    res = drone.move_forward((zDistance+80)/100)
                    print(res)
                    key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)
                print('output',res)
                time.sleep(5)

                # move down
                res = 'none_response'
                while(res != 'ok'):
                    res = drone.move_down(1)
                    cv2.waitKey(5000)

                print(res)


            # marker id = 5
            elif (markerIds == 5):
                if (zDistance > 120):
                    '''
                    先做校正
                    '''
                    # up and down
                    if (yDistance > 15):
                        drone.move_down(0.2)
                        key = cv2.waitKey(5000)
                    elif (yDistance < -15):
                        drone.move_up(0.2)
                        key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)

                    # left and right
                    if (xDistance > 20):
                        drone.move_right(xDistance/100)
                        key = cv2.waitKey(5000)
                    elif (xDistance < -15):
                        drone.move_left(-xDistance/100)
                        key = cv2.waitKey(5000)
                    
                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)
                    
                    # after moving, the distance between the drone and aruco is 70 cm
                    drone.move_forward(0.4)
                    cv2.waitKey(5000)

                    continue
                elif (zDistance > 50):
                    # up and down
                    if (yDistance > 15):
                        drone.move_down(0.2)
                        key = cv2.waitKey(5000)
                    elif (yDistance < -15):
                        drone.move_up(0.2)
                        key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)

                    # left and right
                    if (xDistance > 20):
                        drone.move_right(xDistance/100)
                        key = cv2.waitKey(5000)
                    elif (xDistance < -20):
                        drone.move_left(-xDistance/100)
                        key = cv2.waitKey(5000)

                    if key != -1:
                        drone.keyboard(key)
                        cv2.waitKey(5000)
                    
                    # drone.move_forward(0.2)
                    # cv2.waitKey(5000)

                if (xDistance < 20):
                    drone.move_left(0.4)
                    cv2.waitKey(3000)

                drone.move_forward(0.5)
                cv2.waitKey(5000)


                drone.land()
                cv2.waitKey(3000)


            

        if(len(state) !=0):
            print(state)
        # else:
        #     print('no!')
            
    

        if key != -1:
            drone.keyboard(key)
            cv2.waitKey(5000)
        
        


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()