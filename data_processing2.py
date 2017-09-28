# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:51:13 2017

@author: Toshiharu
"""

import csv
import cv2
import numpy as np


#with open('G:/Documents/GITHUB/CarND-DataSets/data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        samples.append(line)

	
#with open('G:/Documents/GITHUB/CarND-DataSets/left_curve1/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        samples.append(line)
#csv.close(csvfile)
#
#with open('G:/Documents/GITHUB/CarND-DataSets/right_curve1/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        samples.append(line)
#csv.close(csvfile)

	


import sklearn	

#def generator(samples, batch_size=128):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#            images = []
#            angles = []
#            for batch_sample in batch_samples:
#                name = 'G:/Documents/GITHUB/CarND-DataSets/data/IMG/'+batch_sample[0].split('/')[-1]
#                center_image = cv2.imread(name)
#                center_angle = float(batch_sample[3])
#                images.append(center_image)
#                angles.append(center_angle)
#                
#                images.append(cv2.flip(center_image,1))
#                angles.append(center_angle*-1.0)
#
#            # trim image to only see section with road
#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield sklearn.utils.shuffle(X_train, y_train)
            
def generator(filepath, batch_size = 3000):
    
    samples = []
    with open(filepath + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
           
    num_samples = len(samples)
    #print(num_samples)
        #path = filepath + 'IMG/'
        
    while 1: # Loop forever so the generator never terminates
        for offset in range(1, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            
            path = filepath + 'IMG/'
            for batch_sample in batch_samples:
                center_path = path + batch_sample[0].split('/')[-1]
                left_path   = path + batch_sample[1].split('/')[-1]
                right_path  = path + batch_sample[2].split('/')[-1]
                
                img_center = cv2.imread(center_path)
                img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)  # convertion to RGB
                img_left = cv2.imread(left_path)
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB) # convertion to RGB
                img_right = cv2.imread(right_path)
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB) # convertion to RGB
            
                angle_offset = 0.20
                measurement = float(batch_sample[3])
                
                images.append(img_center)
                measurements.append(measurement)
                
                images.append(img_left)
                measurements.append(measurement+angle_offset)
                
                images.append(img_right)
                measurements.append(measurement-angle_offset)
                
                # Augment data by flipping images
                images.append(cv2.flip(img_center,1))
                measurements.append(measurement*-1.0)
    
        
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
