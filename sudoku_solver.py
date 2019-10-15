import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from solver import *
import keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def find_max_area_contour(contours):
    area = -1
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > area:
            index = i
    return index

def warp_coord(pts1):
    pts2 = [[0,0],[0,297],[297,0],[297,297]]
    ref = np.amax(pts1)//2
    pts1 = list(pts1)
    for i in range(len(pts1)):
        if pts1[i][0] < ref:
            if pts1[i][1] < ref:
                print('Top left coordinate:',pts1[i])
                pts2[i] = [0 , 0]
            elif pts1[i][1] > ref:
                print('Bottom left coordinate:',pts1[i])
                pts2[i] = [0, 297]
        elif pts1[i][0] > ref:
            if pts1[i][1] < ref:
                print('Top right coordinate:',pts1[i])
                pts2[i] = [297, 0]
            elif pts1[i][1] > ref:
                print('Bottom right coordinate:',pts1[i])
                pts2[i] = [297, 297]
    return np.float32(pts2)

def find_numbered_squares(squares):
    numbered_squares = list()
    for i in range(81):
        if np.var(numbered_images[i][10:23, 10:23]) > 10000:
            numbered_squares.append(i)
    return numbered_squares

def preprocessing_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #Maybe this is not required
    binary_image = cv2.bitwise_not(thresh, thresh)
    #TODO Dilation necessary?
    kernel = np.ones((1,1), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel)

    lines = cv2.HoughLinesP(dilated_image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    #Use for debugging
    '''
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 1)
    '''
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = find_max_area_contour(contours)

    epsilon = 0.1*cv2.arcLength(contours[index], True)
    approx = cv2.approxPolyDP(contours[index], epsilon, True)

    pts1 = np.float32(approx.reshape(4,2))
    pts2 = warp_coord(pts1)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(dilated_image, M, (297,297))

    plt.imshow(dilated_image)
    plt.show()
    '''
    x_size, y_size = 33,33
    squares = []
    for y in range(1,10):
        for x in range(1,10):
            squares.append(dst[(y-1)*y_size:y*y_size, (x-1)*x_size:x*x_size])
    
    numbered_squares = find_numbered_squares(squares)
    '''





image = cv2.imread('sudoku.jpg')
preprocessing_image(image)
    




