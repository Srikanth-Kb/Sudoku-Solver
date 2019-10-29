import cv2
import numpy as np
from solver import *
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from image_utils import *

model = create_model()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    dilated_image = preprocessing_image(frame)

    approx = find_sudoku_border(dilated_image)

    if len(approx)==4:
        aligned_image = align_sudoku(dilated_image, approx)
        
        square_images_list = obtain_squares_list(aligned_image)

        numbered_squares_list = detect_numbers(square_images_list)

        digits_dict = predict_digits(square_images_list, numbered_squares_list, model)
        
        sudoku_string = create_string(digits_dict)

        answer_dict = solve('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    
        answers_list = list(answer_dict.items())

        aligned_original_image = align_sudoku(frame, approx)

        answered_image = write_answers_on_image(sudoku_string, aligned_original_image, answers_list)

        aligned_answered_image = inverse_perspective(answered_image, approx, frame)

        final_answered_image = cv2.addWeighted(frame, 0.5, aligned_answered_image, 0.5, 0)

        cv2.imshow('Solved Image', final_answered_image)

    else:
        cv2.imshow('Solved Image',frame)
    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







