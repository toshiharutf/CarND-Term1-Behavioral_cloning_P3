# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:51:13 2017

@author: Toshiharu
"""

import csv
import cv2
import numpy as np

images = []
measurements = []

#'G:/Documents/GITHUB/CarND-DataSets/data/driving_log.csv'

def image_generator(filepath):
    lines = []
    with open(filepath + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    for line in lines[1::]:
        path = filepath + 'IMG/'
        
        center_path = path + line[0].split('/')[-1]
        left_path   = path + line[1].split('/')[-1]
        right_path   = path + line[2].split('/')[-1]
        
        img_center = cv2.imread(center_path)
        img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)  # convertion to RGB
        img_left = cv2.imread(left_path)
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB) # convertion to RGB
        img_right = cv2.imread(right_path)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB) # convertion to RGB
    
        angle_offset = 0.20
        measurement = float(line[3])
        
        images.append(img_center)
        measurements.append(measurement)
        
        images.append(img_left)
        measurements.append(measurement+angle_offset)
        
        images.append(img_right)
        measurements.append(measurement-angle_offset)
        
        # Augment data by flipping images
        images.append(cv2.flip(img_center,1))
        measurements.append(measurement*-1.0)
        
    csvfile.close()
    
#image_generator('G:/Documents/GITHUB/CarND-DataSets/data/')
#image_generator('G:/Documents/GITHUB/CarND-DataSets/left_curve2/')
#image_generator('G:/Documents/GITHUB/CarND-DataSets/right_curve2/')
image_generator('G:/Documents/GITHUB/CarND-DataSets/track1cw/')
image_generator('G:/Documents/GITHUB/CarND-DataSets/track1ccw/')

X_train = np.array(images)
y_train = np.array(measurements)
    
training_data = {'features': X_train, 'steering': y_train }    

import pickle

with open('G:/Documents/GITHUB/CarND-DataSets/track1full.p','wb') as handle:
    pickle.dump(training_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

############################################################################ 
#image = cv2.imread('G:/Documents/GITHUB/CarND-DataSets/data/IMG/center_2016_12_01_13_31_13_890.jpg')
#crop_img = image[60:135,: ]
#new_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
#cv2.imshow("Cropped Image", new_img[:,:,2])
#cv2.waitKey(0)
#np.shape(crop_img) # 75x320x3
    




