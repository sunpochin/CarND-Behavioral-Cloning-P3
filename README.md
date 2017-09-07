# Behaviorial Cloning Project

# Overview
---
This is my udacity self driving car project, which tries to train a model to predict which "steering angle" to use according to different situations.

A recording of the simulation could be found here:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4reOzBoT5M/0.jpg)](https://www.youtube.com/watch?v=O4reOzBoT5M)



# Rubric points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.
---

## Files Submitted & Code Quality
---
### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


## Model Architecture and Training Strategy
---
### 1. An appropriate model architecture has been employed

I use a modified network architecture nvidia used for their self driving car, which added some dropouts.
[Link here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

My final model consisted of the following layers:

| Layer             		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         	    	| 64x64x1 Gray-scale normalized image   				|
| Convolution 5x5     	| 2x2 stride, same padding, output depth 24            |
| RELU		        			|												|
| Convolution 5x5	      | 2x2 stride, same padding, output depth 36|
| RELU			        		|												|
| Convolution 5x5	      | 2x2 stride, same padding, output depth 48|
| RELU			        		|												|
| Convolution 3x3	      | 1x1 stride, same padding, output depth 64|
| RELU			        		|												|
| Convolution 3x3	      | 1x1 stride, same padding, output depth 64|
| RELU			        		|												|
| Fully connected		| outputs 1152       							|
| Fully connected		| outputs 1164       							|
| Dropout				| 20% 											|
| Fully connected		| outputs 100       							|
| Dropout				| 20% 											|
| Fully connected		| outputs 50       							|
| Dropout				| 20% 											|
| Fully connected		| outputs 10       							|
| Fully connected		| outputs 1       							|
|						|												|



### 2. Attempts to reduce overfitting in the model
---



### 3. Model parameter tuning
---
#### 1. I use the "default udacity dataset" to reduce changing factors.
#### 2. From some basic EDA I learned a great portion of training data has label steering angle 0, which might cause issues. So I tried to filter out 99% of them using ```random()``` .
#### 3. I use images from all three cameras, and a angle correction of 0.25 .

### 4. Appropriate training data
---


Implementation challenges
---
1. Not enough visualization as debugging. forgot ```image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) . ```
2. keras version doesn't match between my AWS and local machine.
3. To make it driving faster. In ```drive.py``` I can set ```set_speed``` to different values other than 9.
