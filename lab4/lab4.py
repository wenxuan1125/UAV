import cv2
import numpy as np

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

    cap = cv2.VideoCapture(1) #device
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


            cv2.imshow('frame', frame)
            cv2.waitKey(33)

 
    # calculate the parameters
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

    # store the parameters
    f = cv2.FileStorage('parameters.xml', cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()

    print('completed!')


def warping():
    '''
    cap = cv2.VideoCapture(1) #device
    while(True):
        ret, frame = cap.read()
        if ret == True:
            break'''

    # Read the source image.
    src_img = cv2.imread('IU.png')
    # Four corners of the source image
    src_height, src_width, src_channel = src_img.shape  # height width channel
    src_points = np.float32([[0, 0], [src_height - 1, 0], [src_height - 1, src_width - 1], [0, src_width - 1]])


    # Read the destination image.
    dst_img = cv2.imread('warp.jpg')    
    # Four pasted corners in the destination image.
    dst_points = np.float32([[202, 245],[640, 260],[763, 432],[244, 462]])
    dst_height, dst_width, dst_channel = dst_img.shape

    mat = cv2.getPerspectiveTransform(src_points, dst_points)
    inv_mat = np.linalg.inv(mat)
    pasted_img = np.zeros(dst_img.shape, np.uint8)



    for i in range(dst_height):
        for j in range(dst_width):

            
            pos = np.array([i, j, 1]).reshape(3, -1)    # reshape(m, n) reshape the array to a m*n matrix
            mapping_pos = np.dot(inv_mat, pos)
            mapping_pos = mapping_pos/mapping_pos[2]
            
            
            if (mapping_pos[0] >= 0 and mapping_pos[0] < src_height and mapping_pos[1] >= 0 and mapping_pos[1] < src_width):
                
                up_left_i = int(mapping_pos[0])
                up_left_j = int(mapping_pos[1])

                up_left = src_img[up_left_i, up_left_j]
                down_left = src_img[min(up_left_i + 1, src_height-1), up_left_j]
                down_right = src_img[min(up_left_i + 1, src_height-1), min(up_left_j + 1, src_width-1)]
                up_right = src_img[up_left_i, min(up_left_j + 1, src_width-1)]

                
                        
                #interpolation for down_left and down_right
                temp1 = (up_left_j + 1 - mapping_pos[1]) * down_left + (mapping_pos[1] - up_left_j) * down_right
                #interpolation for up_left and up_right
                temp2 = (up_left_j + 1 - mapping_pos[1]) * up_left + (mapping_pos[1] - up_left_j) * up_right

                #interpolation for up and down
                pasted_img[i, j] = (up_left_i + 1 - mapping_pos[0]) * temp2 + (mapping_pos[0] - up_left_i) * temp1
            else:
                pasted_img[i, j]=dst_img[i, j]



    # Display images
    cv2.imshow("Pasted Image", pasted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


warping()
    