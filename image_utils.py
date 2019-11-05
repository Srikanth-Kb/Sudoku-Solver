import cv2 
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def find_max_area_contour(contours):
    area = -1
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > area:
            area = cv2.contourArea(contours[i])
            index = i
    return index

def warp_coord(pts1):
    pts2 = [[0,0],[0,297],[297,0],[297,297]]
    ref = np.amax(pts1)//2
    pts1 = list(pts1)
    for i in range(len(pts1)):
        if pts1[i][0] < ref:
            if pts1[i][1] < ref:
                #print('Top left coordinate:',pts1[i])
                pts2[i] = [0 , 0]
            elif pts1[i][1] > ref:
               # print('Bottom left coordinate:',pts1[i])
                pts2[i] = [0, 297]
        elif pts1[i][0] > ref:
            if pts1[i][1] < ref:
                #print('Top right coordinate:',pts1[i])
                pts2[i] = [297, 0]
            elif pts1[i][1] > ref:
                #print('Bottom right coordinate:',pts1[i])
                pts2[i] = [297, 297]
    return np.float32(pts2)

def preprocessing_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #Maybe this is not required
    binary_image = cv2.bitwise_not(thresh, thresh)
    #TODO Dilation necessary?
    kernel = np.ones((1,1), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel)

    return dilated_image

def find_sudoku_border(image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        index = find_max_area_contour(contours)
        epsilon = 0.1*cv2.arcLength(contours[index], True)
        approx = cv2.approxPolyDP(contours[index], epsilon, True)

        return approx

def align_sudoku(image, approx):
        try:
            pts1 = np.float32(approx.reshape(4,2))
            pts2 = warp_coord(pts1)

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(image, M, (297,297))
        except:
            return image
        else:
            return dst

def obtain_squares_list(image):
        x_size, y_size = 33,33
        squares = []
        for y in range(1,10):
            for x in range(1,10):
                squares.append(image[(y-1)*y_size:y*y_size, (x-1)*x_size:x*x_size])
        return squares

def detect_numbers(squares):
    numbered_squares = list()
    for i in range(81):
        if np.var(squares[i][10:23, 10:23]) > 10000:
            numbered_squares.append(i)
    return numbered_squares

def create_model():
        input_shape = (28,28,1)
        num_classes = 10
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
        model = load_model('mnist_model.h5')

        return model

def predict_digits(squares, numbered_squares, model):
        my_dict = {}
        for index in numbered_squares:
            buff = squares[index][3:30,3:30]
            buff = cv2.resize(buff, (28,28))
            buff = buff.reshape(1,28,28,1)
            label = model.predict_classes(buff)
            my_dict[index] = label[0]
            #print('Block number:',index, '\t','Predicted number:',label)
            #plt.subplot(9,9,index+1), plt.imshow(squares[index][3:30,3:30],cmap='gray')
            #plt.title(label)
        return my_dict

def create_string(detected_numbers_dictionary):
        sudoku = ['0' for i in range(81)]
        for key, value in detected_numbers_dictionary.items():
            sudoku[key] = str(value)
        grid1 = ''.join(sudoku)

        return grid1

def write_to_image(image, item):
    row = item[0][0]
    column = item[0][1]
    value = item[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, value, (9+33*(int(column)-1), 25+33*(ord(row)-65)), font, 0.7, (0,255,0),2, cv2.LINE_AA)
    return image

def write_answers_on_image(grid1, image, answers):
    for i in range(81):
        if grid1[i]=='0' or grid1[i]=='.':
            frame = write_to_image(image, answers[i] )
    return frame

def inverse_perspective(answer_image, approx, image):
    pts1 = np.float32(approx.reshape(4,2))
    pts2 = warp_coord(pts1)

    N = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(answer_image, N, image.shape[1::-1])

    return dst