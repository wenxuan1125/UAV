import cv2
import numpy as np
import time

def preprocess(img_path1,img_path2):
    img_1 = cv2.imread(img_path1)
    img_2 = cv2.imread(img_path2)
    # img_1 = cv2.resize(img_1, (400, 540))
    # img_2 = cv2.resize(img_2, (400, 540))

    return img_1, img_2

def sift_func(img_1,img_2):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    start = time.time()
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    psd_kp1, psd_des1 = sift.detectAndCompute(gray_1, None)
    psd_kp2, psd_des2 = sift.detectAndCompute(gray_2, None)

    end = time.time()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)

    # Need to draw only good matches, so create a list
    goodMatch = list()

    # Apply ratio test
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，
        # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)

    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    img_out = cv2.drawMatchesKnn(img_1, psd_kp1,
                                 img_2, psd_kp2,
                                 goodMatch, None, flags=2)
    return img_out, end - start

def surf_func(img_1,img_2):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    start = time.time()

    surf = cv2.xfeatures2d.SURF_create()
    psd_kp1, psd_des1 = surf.detectAndCompute(gray_1, None)
    psd_kp2, psd_des2 = surf.detectAndCompute(gray_2, None)

    end = time.time()

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)
    goodMatch = list()
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，
        # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)

    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    img_out = cv2.drawMatchesKnn(img_1, psd_kp1,
                                 img_2, psd_kp2,
                                 goodMatch, None, flags=2)
    return img_out, end - start

def orb_func(img_1, img_2):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    start = time.time()

    orb = cv2.ORB_create()
    psd_kp1, psd_des1 = orb.detectAndCompute(gray_1, None)
    psd_kp2, psd_des2 = orb.detectAndCompute(gray_2, None)

    end = time.time()

    # Flann特征匹配
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)
    goodMatch = list()
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，
        # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.6*n.distance:
            goodMatch.append(m)

    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    img_out = cv2.drawMatchesKnn(img_1, psd_kp1,
                                 img_2, psd_kp2,
                                 goodMatch, None, flags=2)
    return img_out, end - start


if __name__ == '__main__':
    img_path1, img_path2 = 'D:/C++/C-project/uav/lab11/ele1.jpg', 'D:/C++/C-project/uav/lab11/ele2.jpg'
    img_1, img_2 = preprocess(img_path1, img_path2)
    sift_img, time1 = sift_func(img_1, img_2)
    surf_img, time2 = surf_func(img_1, img_2)
    orb_img, time3 = orb_func(img_1, img_2)
    cv2.imwrite('sift_image.jpg', sift_img)
    cv2.imwrite('surf_image.jpg', surf_img)
    cv2.imwrite('orb_image.jpg', orb_img)

    print('SIFT time: ', str(time1))
    print('SURF time: ', str(time2))
    print('ORB time: ', str(time3))

    cv2.imshow('sift image', sift_img)
    cv2.imshow('surf image', surf_img)
    cv2.imshow('orb image', orb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()