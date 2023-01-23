import numpy as np
import cv2

def bilinear( img_name, magnification):
    #img_name = 'IU.png'
    #magnification = 3

    img = cv2.imread(img_name)
    size = img.shape  # height width channel
    bilinear_img = np.zeros((size[0] * magnification, size[1] * magnification, size[2]), np.uint8)
    bilinear_size = bilinear_img.shape  # height width channel

    for i in range(bilinear_size[0]):
        for j in range(bilinear_size[1]):
            up_left_i = int(i / magnification)
            up_left_j = int(j / magnification)

            up_left = img[up_left_i, up_left_j]
            down_left = img[min(up_left_i + 1, size[0]-1), up_left_j]
            down_right = img[min(up_left_i + 1, size[0]-1), min(up_left_j + 1, size[1]-1)]
            up_right = img[up_left_i, min(up_left_j + 1, size[1]-1)]

            '''
            # zero padding
            if (up_left_i < size[0] - 1):

                up_left = img[up_left_i, up_left_j]
                down_left = img[up_left_i + 1, up_left_j]
                
                if (up_left_j < size[1] - 1):
                    down_right = img[up_left_i + 1, up_left_j + 1]
                    up_right = img[up_left_i, up_left_j + 1]
                else:
                    down_right = 0
                    up_right = 0
            else:

                up_left = img[up_left_i, up_left_j]

                if (up_left_j < size[1] - 1):
                    down_left = 0
                    down_right = 0
                    up_right = img[up_left_i, up_left_j + 1]
                else:
                    down_left = 0
                    down_right = 0
                    up_right = 0'''
                    
            #interpolation for down_left and down_right
            temp1 = (up_left_j + 1 - j / magnification) * down_left + (j / magnification - up_left_j) * down_right
            #interpolation for up_left and up_right
            temp2 = (up_left_j + 1 - j / magnification) * up_left + (j / magnification - up_left_j) * up_right

            #interpolation for up and down
            bilinear_img[i, j] = (up_left_i + 1 - i / magnification) * temp2 + (i / magnification - up_left_i) * temp1
        
    #cv2.imshow('original image', img)
    cv2.imshow('bilinear image', bilinear_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nearest_neighbor():
    img = cv2.imread('IU.png')
    size = img.shape  # height width channel
    nn_img = np.zeros((size[0] * 3, size[1] * 3, size[2]), np.uint8)
    nn_size = nn_img.shape  # height width channel

    for i in range(nn_size[0]):
        for j in range(nn_size[1]):
            nn_img[i,j] = img[int(i/3),int(j/3)]
        
    #cv2.imshow('original image', img)
    cv2.imshow('nearest neighbor image', nn_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate():
    img = cv2.imread('curry.jpg')
    size = img.shape  # height width channel
    ''' 
    #resize
    img = cv2.resize(img, (int(size[0]/2), int(size[1]/2)))
    size = img.shape  # height width channel'''
    rotate_img = np.zeros((size[1], size[0], size[2]), np.uint8)

    for i in range(size[0]):
        rotate_img[:,i] = img[i,::-1]
    '''        
    #resize
    cv2.namedWindow('rotate image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rotate image', (size[1], size[0]))'''
    cv2.imshow('rotate image', rotate_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def flip():
    img = cv2.imread('kobe.jpg')
    size = img.shape  # height width channel
    flip_img = np.zeros(size, np.uint8)
    #flip_img = img.copy()

    for i in range(size[1]):
        flip_img[:,size[1]-i-1] = img[:,i] 

    cv2.imshow('flip image', flip_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

flip()
rotate()
nearest_neighbor()
bilinear('IU.png', 3)

