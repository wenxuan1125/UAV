import numpy as np
import cv2
from disjoint_set import DisjointSet

cap = cv2.VideoCapture('vtest.avi')
backSub = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened()):
    ret, frame = cap.read()
    
    
    fgmask = backSub.apply(frame)
    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)

    nmask_size = nmask.shape  # height width

    label_id = 1   # recording the number of the labels assigned    
    label_img = np.zeros(nmask_size)  # recording the connected component image
    ds = DisjointSet()
    
    # first pass
    for i in range(nmask_size[0]):
        for j in range(nmask_size[1]):

            if (nmask[i, j] == 255):

                if (i == 0 and j == 0):
                    label_img[i, j] = int(label_id)

                    label_id = label_id + 1
                elif (i == 0):
                    if (label_img[i, j - 1] == 0):
                        label_img[i, j] = int(label_id)

                        label_id = label_id + 1
                    else:
                        label_img[i, j] = label_img[i, j - 1]
                elif (j == 0):
                    if (label_img[i - 1, j] == 0):
                        label_img[i, j] = int(label_id)

                        label_id = label_id + 1
                    else:
                        label_img[i, j] = label_img[i - 1, j]
                else:

                    # both the above and the left pixel are not labeled => new label 
                    if (label_img[i - 1, j] == 0 and label_img[i, j - 1] == 0):
                        label_img[i, j] = int(label_id)
                        
                        label_id = label_id + 1
                    # both the above and the left pixel are labeled => conflict
                    # union these two labels
                    elif (label_img[i - 1, j] != 0 and label_img[i, j - 1] != 0):
                    
                        label_img[i, j] = min(label_img[i - 1, j], label_img[i, j - 1])
                        ds.union(int(label_img[i - 1, j]), int(label_img[i, j - 1]))
                    # only one of the above and the left pixel are labeled
                    else:
                        label_img[i, j] = max(label_img[i - 1, j], label_img[i, j - 1])


    label_area = np.zeros(label_id)
    # coordinate
    label_up = np.empty(label_id)
    label_left = np.empty(label_id)
    label_right = np.empty(label_id)
    label_down = np.empty(label_id)
    
    for i in range(label_id):
        # initialize
        label_up[i] = nmask_size[0] + 1
        label_left[i] = nmask_size[1] + 1
        label_right[i] = -1
        label_down[i] = -1

    
    for i in range(nmask_size[0]):
        for j in range(nmask_size[1]):

            if (label_img[i, j] != 0):

                if (label_img[i, j] != ds.find(label_img[i, j])):
                    label_img[i, j] = ds.find(label_img[i, j])
                
                label_area[int(label_img[i, j])] = label_area[int(label_img[i, j])] + 1
                label_up[int(label_img[i, j])] = min(label_up[int(label_img[i, j])], i)
                label_down[int(label_img[i, j])] = max(label_down[int(label_img[i, j])], i)
                label_left[int(label_img[i, j])] = min(label_left[int(label_img[i, j])], j)
                label_right[int(label_img[i, j])] = max(label_right[int(label_img[i, j])], j)

    T = 30
    
    for i in range(label_id):
        if (label_area[i] > T):
            cv2.rectangle( frame, (int(label_left[i]), int(label_up[i])), (int(label_right[i]), int(label_down[i])), (255,0, 0), 2)


    cv2.imshow("frame", frame)
    cv2.waitKey(33)


            

    


    
