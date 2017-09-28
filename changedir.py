# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:58:41 2017

@author: Toshiharu

This script just change the filepath, so it can be compatible with the
image processing script

"""

import csv
import numpy as np
import pandas as pd

samples = []

with open('G:/Documents/GITHUB/CarND-DataSets/track2ccw/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
    	samples.append(line)


for line in samples[1::]:
    for i in range(3):
        line[i] = 'IMG/' + line[i].split('\\')[-1]
csvfile.close()     
	#samples.append(line)
        
#print(samples)

with open('G:/Documents/GITHUB/CarND-DataSets/track2ccw/driving_log.csv','w',newline='') as csvfile2:
    writer = csv.writer(csvfile2,delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(samples)
csvfile2.close()