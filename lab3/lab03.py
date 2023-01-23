import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')
backSub = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened()):
    ret, frame = cap.read()
    
    
    fgmask = backSub.apply(frame)
    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)

    nmask_size = nmask.shape  # height width


    label_list = [0]    # recording what label the label[i] maps to
    label_id = 1        
    label_img = np.zeros(nmask_size)    # recording the connected component image
    
    # first pass
    for i in range(nmask_size[0]):
        for j in range(nmask_size[1]):

            if (nmask[i, j] == 255):

                if (i == 0 and j == 0):
                    label_img[i, j] = int(label_id)

                    
                    label_list.append(label_id)
                    label_id = label_id + 1
                elif (i == 0):
                    if (label_img[i, j - 1] == 0):
                        label_img[i, j] = int(label_id)
                        
                        label_list.append(label_id)
                        label_id = label_id + 1
                    else:
                        label_img[i, j] = label_img[i, j - 1]
                elif (j == 0):
                    if (label_img[i - 1, j] == 0):
                        label_img[i, j] = int(label_id)
                        label_list.append(label_id)
                        label_id = label_id + 1
                    else:
                        label_img[i, j] = label_img[i - 1, j]
                else:

                    # both the above and the left pixel are not labeled => new label 
                    if (label_img[i - 1, j] == 0 and label_img[i, j - 1] == 0):
                        label_img[i, j] = int(label_id)
                        
                        label_list.append(label_id)
                        label_id = label_id + 1
                    # both the above and the left pixel are labeled => conflict
                    # the label of this pixel is set to the smaller label
                    # the value of the larger label in label_list is mapped to that of the smaller label 
                    elif (label_img[i - 1, j] != 0 and label_img[i, j - 1] != 0):
                    
                        label_img[i, j] = min(label_img[i - 1, j], label_img[i, j - 1])
                        label_list[int(max(label_img[i - 1, j], label_img[i, j - 1]))] = min(label_list[int(label_img[i - 1, j])], label_list[int(label_img[i, j-1])])
                    # only one of the above and the left pixel are labeled
                    else:
                        label_img[i, j] = max(label_img[i - 1, j], label_img[i, j - 1])


    label_area = np.zeros(len(label_list))
    # coordinate
    label_up = np.empty(len(label_list))
    label_left = np.empty(len(label_list))
    label_right = np.empty(len(label_list))
    label_down = np.empty(len(label_list))
    
    for i in range(len(label_list)):
        # ensure that the value 
        if (label_list[i] != i):
            label_list[i] = label_list[int(label_list[i])]
        
        # initialize
        label_up[i] = nmask_size[0] + 1
        label_left[i] = nmask_size[1] + 1
        label_right[i] = -1
        label_down[i] = -1

    
    for i in range(nmask_size[0]):
        for j in range(nmask_size[1]):

            if (label_img[i, j] != 0):

                if (label_img[i, j] != label_list[int(label_img[i, j])]):
                    label_img[i, j] = label_list[int(label_img[i, j])]
                
                label_area[int(label_img[i, j])] = label_area[int(label_img[i, j])] + 1
                label_up[int(label_img[i, j])] = min(label_up[int(label_img[i, j])], i)
                label_down[int(label_img[i, j])] = max(label_down[int(label_img[i, j])], i)
                label_left[int(label_img[i, j])] = min(label_left[int(label_img[i, j])], j)
                label_right[int(label_img[i, j])] = max(label_right[int(label_img[i, j])], j)

    T = 30
    
    for i in range(len(label_list)):
        if (label_area[i] > T):
            cv2.rectangle( frame, (int(label_left[i]), int(label_up[i])), (int(label_right[i]), int(label_down[i])), (255,0, 0), 2)


    cv2.imshow("frame", frame)
    cv2.waitKey(33)


            

    


    
