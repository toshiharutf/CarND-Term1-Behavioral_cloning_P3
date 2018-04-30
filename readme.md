# **Behavioral Cloning Project** 

---

The objective of this project was to design an algorithm using deep learning, which can "clone" a human driver behavior and drive a simulated vehicle by itself. In order to train the network, several laps of good driving behavior were recorded. For this project, good driving behavior implies driving in the center of the lane as smooth as possible. The recorded data was also augmented with image processing, and some geometrical assumptions to create more disperse data, so that the model generalize for more scenenarios. The final trained model was able to complete several laps, while maintaining inside the bound of the lane.

**The goals / steps of this project are the following:**

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test  the model in the track. It should drive around track one without leaving the road.
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center camera"
[image2]: ./images/left.jpg   "Left camera"
[image3]: ./images/right.jpg  "Right camera"
[image4]: ./images/simulator.jpg  "Simulator view"
[image5]: ./images/flip.jpg  "Fipped image"
[track2_1]: ./images/track2_s1.jpg "Sample of track 2"
[hist1]: ./images/before_data_aug.png "Flipped Image"
[hist2]: ./images/after_data_aug.png "Flipped Image"

[gif1_1]: ./videos/track1.gif
[gif1_2]: ./videos/track1_offtrack.gif

[gif2_1]: ./videos/track2.gif

## Files of this project
* model.py : Implements the deep learning network, load the training data, and finally train the network.
* fine_tuning.py : It loads previously tuned weights and retrain the network using new data, like in those parts in which the algorithm would drive the car outside the lane!
* drive.py : Loads the trained weights of the networks and drive autonomously in the simulator.
* track1_weights.h5 : weights trained with data of the track 1
* track2_weights.h5 : weights trained with data of the track 2
* videos/track1_fpv.mp4: Video recorded of test on the first track, from the central camera of the car
* videos/track2_fpv.mp4: Video recorded of test on the second track, from the central camera of the car
* P3_writeup.md : The document you are reading right now.

## Deep neural network for end-to-end autonomous driving

### Preprocessing

Normalization and image cropping layers are added inside the Keras neural network. The normalization layer is actually quite simple, but effective. It just divide each of element by 255 (maximum value of each pixel). Next, the top of the image, over the horizon of the road, and the bottom in which the hood of the car appears, are cropped. This allows the network to concentrate almost exclusively on the road. Other kind of image processing (color filtering, edge detection, etc) were not applied, to test the neural network performance in filtering unrelevant elements of the image.

### Model Architecture
The neural network used in this project was based on the **NVIDIA** convolutional network. This architecture was specifically designed for end-to-end driving capability using as input information from the camera. Also, it has positive feedback from previous udacity students for the same type of problem.

The network in this project has:


-  1 normalization layer
-  1 Cropping layer
-  5 convolutional layers with relu activation function
-  1 fully connected layer with elu activation function
-  Dropout at 50%
-  1 fully connected layer with elu activation function
-  Dropout at 30%
-  1 fully connected layer with elu activation function
-  Dropout at 10%
-  1 output with a tanh activation function

**Note**: For more details, refer to the file "model.py"

### Optimizer and early stop options
The chosen optimizer is Adams. It is configured with a learning rate decay of 0.7, which allowed a smoother driving experience.
Also, a minimum delta of 0.00009 is imposed as an early stop option. The minimum square error "mse" is used as the measure of deviation.


### Approach to reduce overfitting
For this project, dropout techniques were the principal way to reduce overfitting. Dropout was applied to fully connected layers, except for the output one. Also, early stopping options, data augmentation, and fine tunning with new training data could also contribute to avoid overfitting. Check model.py lines 37 to 49.


---

## Training data

Data were collected by manually driving the car in the provided tracks of the Udacity simulator. The resulting data are series of images, from the center, left, and right inside cameras of the car, plus the steering angles, which varies from -1 to 1, where -1 means full left, and 1 full right. After running a data augmentation algorithm, all the resulting images and measurements were stored in a pickle file, for easier manipulation. Although it can take around a minute to load the data to memory, the GPU can train the network very efficiently and fast.

![alt text][image4] 

### Track 1

The first track is an all flat terrain with well defined lanes on most parts. In order to obtain good data, the car was carefully driven with speeds between 2 to 10 mph, at the center of the lane. Two laps were made in each clockwise, and counterclockwise directions. The captured images simulate a first-person-view (FPV) from cameras looking from the center, left and right sides of the car. This means that captured images will not look the same as when playing in the simulator.

**Center Camera**

![alt text][image1] 

**Left Camera**

![alt text][image2]

**Right Camera**

![alt text][image3]

### Data augmentation
The obtained training data was manipulated to artificially increase the number of samples, without recording new data. Two operations were made: horizontal flipping, and simulating off track situations with left and right cameras.

#### a. Horizontal flip
The first manipulation was horizontal flipping. The corresponding steering angle was multiplied by -1 accordingly. This duplicate the number of data, and helps generalizing the model for more driving situations.

 **Original image**

![alt text][image1]

**Flipped image**

![alt text][image5]

	# Python code for flipping images and angles    
	images.append(cv2.flip(img_center,1))
    measurements.append(measurement*-1.0)

#### b. Simulating going off track
In the first approach, the proposed network was trained only using the central camera images. The performance with only this data was not acceptable. For the second approach, in order to teach the model how to recover when the car deviates from the lane center, the car was manually driven to the sides, and then steered back to the center. However, this new data decreased the performance of the model, instead of improving it.

Finally, in the third approach, images from the left and right cameras are used to simulate deviation recovery. For the first track, this recovery angle is manually tuned to 0.20 (over a maximum of 1.0). Higher values show unwanted oscilatory response in the final trained model (speed of 30 mph). The simulated approach provide significant improvements, without the need of recording additional data.

	# Corresponding python code for the offtrack correction
	images.append(img_left)
    measurements.append(measurement+angle_offset)
    
    images.append(img_right)
    measurements.append(measurement-angle_offset) 


Data augmentation also helps the model generalizes, since in most of the track, the car drive straight and most of the steering angles recorded lay around 0.0º, as seen the next histogram.

 **Histogram before data augmentation**

![alt text][hist1] 


After running the data augmentation function, the data is more disperse, and noy only concentrated around 0.0, as seen in the next histogram

 **Histogram after data augmentation**

![alt text][hist2] 

#### c. Fine tuning of the neural network
After the model was trained with the mentioned data set, the algorithm could drive fairly well, except from some segments of the track, as seen in the next video.

![alt text][gif1_2] 

[Going off track - Link to video](https://youtu.be/sh0G-oNnqAA?t=48s)

So the neural network was retrained with new data of only on specific segments of the road, using the already trained weights as the starting point. The code can be seen in fine_tuning.py.
 This approach is more efficient, and less time consuming than retraining the whole model. After retraining, the model could pass the problematic segment without problems.


### Track 2

The second track presented more challenging siutations compared to the first one, as can be seen in the image below. 

![alt text][track2_1]

Curves on this track had shorter turning radius. Also, most parts in the track presented negative, and positive slopes. Uphill segments were significantlly more difficult, since they decreased the visibility of the ahead track. Also, those road inclination changes mean 2D image deformation, thus increasing the difficulty of training the neural network.
Two complete laps were carefully recorded. In a first attemp, the car was driven inside the right path of the lane. However, the network was not able to drive on this side of the lane, and finished going off track.
In the second approach, the car was driven in the center of the road, using the separation lane as guidelines. The wider safety range on the borders allowed the car to complete the track.

The model trained with data of the first track was not able to drive correctly on the second track. For that reason, whole model was retrained from scratch using new data.

### Data augmentation

#### a. Simulating going off track and fine tuning
The horizontal flipping is identical as the in the first track, so the explanation is omitted. However, in the second track, the augmented data, specially the one from simulating going off track played an important role in the performance of the algorithm. At the begginning, the approach used in the first track was used. An angle of 0.20 was used for simulating recovery situations. Also, small segments of the track, in which the neural network has problems, were driven one more time to obtain additional data. However,  after retraining the network with this new tuning data, *the model forgot how drive on other segments of the track*.

A second approach gave the off track simulation more priority. Concretely, more aggressive recovery angles were used. The angle was progressively incresed, starting from 0.20, with steps of 0.10. The neural network was trained from scratch with each new recovery angle. Finally, **an angle of 0.70** allowed the model to complete the whole second track without going outside the track. This high recovery angle respond to the lower turning radius and constantly changing slopes of the track. As a final remark for this section, no fine tuning was used for training the neural network in the second track.



## Training the neural network
The neural network was trained with different training sets. The number of epochs was set to 20, but due to the early stops options, the training stopped before this limit was reached, in some occasions. The PC setup had a Core i5-2500K with an Nvidia GTX1060 with 6GB of RAM. Each epoch took around 60s to complete.


## Testing the model on the simulator
Finally, the trained model is tested in both tracks of the Udacity driving simulator.

### Track 1

![alt text][gif1_1] 

[Track 1 - Link to video](https://youtu.be/AOnOh7KIvpI)

### Track 2

![alt text][gif2_1] 

[Track 2 - Link to video](https://youtu.be/N7J5HFT_vEg)


## Conclusion and future work
In this project, a deep neural network for end-to-end autonomous driving was coded, trained, and simulated. The input data were images from the central camera of the simulated car. After training and fine tuning, the algorithm was succesfully tested on two simulated tracks. 
Future work may include reducing the shakiness on the steering. This could problaby be fixed with better data and better adjusment in the optimizer. Also, trying other architectures like VGG16, or commaai may improve the perfomance.
For the second track, a more challenging task could be designing a neural network that drives only on the right or left side of the lane. 
Also, in the present work, the task of image filtering had relied solely in the neural network. Specialized image filtering before the neural network may increase the performance of the algorithm. This could allow to design a unique neural network, which can drive in both tracks, without modifications.


## How to run the programm
Requirements:
Keras min 2.0.6
python 3.5
Udacity simulator for the first term. [download here](https://github.com/udacity/self-driving-car-sim)

To run the driving algorithm using my trained weights:

*First track*
python drive.py track1_weights.h5

*Second track*
python drive.py track2_weights.h5

Then, run the Udacity simulator in the corresponding track.

Since the second track is very complex, the NN had to be retrained. However, the initial weight values were obtained from the first track, so it took less time to train the second one.
