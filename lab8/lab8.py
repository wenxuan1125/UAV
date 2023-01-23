import cv2
import dlib
import numpy as np
import time

device = 0

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

    cap = cv2.VideoCapture(device) # 0 is laptop, 1 is webcam
    while(True):
        ret, frame = cap.read()
        #ret is True if read() successed

        if ret == True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret2, corners = cv2.findChessboardCorners(gray_frame, (row, column), None)

            # If found, add object points, image points (after refining them)
            if ret2 == True:
            
                
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
    # f = cv2.FileStorage('parameters' + str(device) + '.xml', cv2.FILE_STORAGE_WRITE)
    f = cv2.FileStorage('D:/C++/C-project/uav/lab8/parameters0.xml', cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()

    print('calibration completed!')

def main():

    # calibration and read the parameters
    # camera_calibration()
    # cv_file = cv2.FileStorage("parameters" + str(device) + ".xml", cv2.FILE_STORAGE_READ)
    cv_file = cv2.FileStorage("D:/C++/C-project/uav/lab8/parameters0.xml", cv2.FILE_STORAGE_READ)
    intrinsic = cv_file.getNode("intrinsic").mat()
    distortion = cv_file.getNode("distortion").mat()
    cv_file.release()


    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    # initialize the HOG+SVM person detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # acquire the default face detector
    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0) # 0 is laptop, 1 is webcam
    while(True):

        ret, frame = cap.read()

        # ret is True if read() successed
        if ret == True:

            # person detect
            person_rects, weights = hog.detectMultiScale(frame, 
                                                winStride=(4, 4),
                                                padding=(8, 8),
                                                scale=1.25,
                                                useMeanshiftGrouping=False)

            for (x, y, w, h) in person_rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # pose estimation
                imgpoints= np.array([(x, y), (x, y+h), (x+w, y+h), (x+w, y)], np.float32)
                objpoints = np.array([(0, 0, 0), (200, 0, 0), (200, 50, 0), (0, 50, 0)], np.float32)
                retval, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, intrinsic, distortion)

                # cv2.putText( img, text to show, coordinate, font, font size, color, line width, line type )
                cv2.putText(frame, str(tvec[2]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,\
                0.4, (0, 255, 0), 1, cv2.LINE_AA)
                print(tvec[2], w, h)

                
            
            # face detect
            face_rects = detector(frame, 1)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


                # pose estimation
                imgpoints= np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], np.float32)
                objpoints = np.array([(0, 0, 0), (0, 18, 0), (18, 18, 0), (18, 0, 0)], np.float32)
                retval, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, intrinsic, distortion)

                # cv2.putText( img, text to show, coordinate, font, font size, color, line width, line type )
                cv2.putText(frame, str(tvec[2]), (x2, y2+10), cv2.FONT_HERSHEY_SIMPLEX,\
                0.4, (255, 0, 0), 1, cv2.LINE_AA)

            

            #print()


            


            cv2.imshow('frame', frame)
            key = cv2.waitKey(33)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()