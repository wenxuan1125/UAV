import numpy as np
import cv2
import math

def lab2_1():
    img = cv2.imread('mj.tif')
    size = img.shape    # img.shape = [height, width, channel]
    height = img.shape[0]  
    width = img.shape[0]
    he_img = np.zeros(size, np.uint8)
    probability = np.zeros(256)
    acumulated_probability = np.zeros(256)
    
    intensity_output = np.zeros(256)

    
    # culculate the number of pixels in each intensity 
    for i in range(size[0]):
        for j in range(size[1]):
            probability[img[i, j, 0]] += 1
            
    probability = probability / (size[0] * size[1])

    acumulated_probability[0] = probability[0]
    for i in range(1, 256):
        acumulated_probability[i] = acumulated_probability[i - 1] + probability[i]

    for i in range(256):
        intensity_output[i] = int(acumulated_probability[i] * 255)
        
        
        
    for i in range(size[0]):
        for j in range(size[1]):
            he_img[i, j,:] = intensity_output[img[i, j, 0]]
            
    cv2.imshow('my image', he_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lab2_2():
    img = cv2.imread('input.jpg')
    size = img.shape    # img.shape = [height, width, channel]
    height = img.shape[0]  
    width = img.shape[0]
    he_img = np.zeros(size, np.uint8)
    output_img = np.zeros(size, np.uint8)
    probability = np.zeros(256)
    acumulated_probability = np.zeros(256)
    intensity_output = np.zeros(256)
    intensity = np.zeros(256)
    intensity_num = np.zeros(256)
    
    # culculate the number of pixels in each intensity 
    for i in range(size[0]):
        for j in range(size[1]):
            intensity_num[img[i, j, 0]] += 1
            
    probability = intensity_num / (size[0] * size[1])

    for i in range(256):
        intensity[i] = i

    # find the threshold with max var between groups
    threshold = 0
    max_var = 0
    
    # initial condition
    num_left = 0
    num_right = sum(intensity_num[0:])
    mu_left = 0
    mu_right = sum(intensity[threshold:]*intensity_num[threshold:])/sum(intensity_num[threshold:])
   
    for i in range(256):
        var = num_left * num_right * pow((mu_left - mu_right), 2)
        
        if (var > max_var):
            threshold = i
            max_var = var
            
        if (i != 255):  
            mu_left = (mu_left * num_left + intensity_num[i] * i) / (num_left + intensity_num[i])
            mu_right = (mu_right * num_right - intensity_num[i] * i)/(num_right - intensity_num[i])
            num_left += intensity_num[i]
            num_right -= intensity_num[i]
            
    for i in range(size[0]):
        for j in range(size[1]):
            if (img[i, j, 0] < threshold):
                output_img[i, j,:] = 0
            else:
                output_img[i, j,:] = 255
    
    cv2.imshow('my image', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


lab2_1()
lab2_2()
