# Behaviorial Cloning Project

[vis-image1]: ./pics/bins100.png "bins 100"
[vis-image2]: ./pics/bins500.png "bins 500"
[vis-image3]: ./pics/cnn-architecture-624x890.png "nvidia arch."


[vis-aug-1]: ./pics/center_2016_12_01_13_31_13_786.jpg "center"
[vis-aug-2]: ./pics/left_2016_12_01_13_31_13_786.jpg "left"
[vis-aug-3]: ./pics/right_2016_12_01_13_31_13_786.jpg "right"
[vis-aug-4]: ./pics/cropped.png "cropped"
[vis-aug-5]: ./pics/resized.png "resized"
[vis-aug-6]: ./pics/flipped-1.png "flipped"
[vis-aug-7]: ./pics/normalized-1.png "normalized-1.png"
[vis-aug-8]: ./pics/normalized-2.png "normalized-2.png"


# Overview

This is my udacity self driving car project, which tries to train a model to predict which "steering angle" to use according to different situations.

A recording of the simulation could be found here:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4reOzBoT5M/0.jpg)](https://www.youtube.com/watch?v=O4reOzBoT5M)

I use default udacity training set, started with LeNet-5 architecture and the nvidia one mentioned from previous lessons, both didn't work.

Then I tried tuning with various techniques learned from the class forum, including:
1. Cropping the images.
2. Resizing the images.
3. Normalizing the images.
4. Flippping 50 percent of the images horizontally.
5. Adding dropouts to avoid over fitting.
6. Remove 99% of steering == 0 datas.

Finally I got a working model.


<br>

# Rubric points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* readme.md summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


## Model Architecture and Training Strategy

### * An appropriate model architecture has been employed

I use the network architecture nvidia used for their self driving car, which added some dropouts.
[Link here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

![vis-image3]

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



### * Attempts to reduce overfitting in the model

1. I used 20% of training data as validation set. (``` train_test_split(samples, test_size=0.2) ``` )
2. I added 3 dropout layers which seem to reduce the wobbling driving.
3. I learned a technique to turned horizontally 50% of image, which fixed the car running into the sandy ground area. (```def flip_50_percent_image```)


### * Model parameter tuning

1. From some basic EDA I learned a great portion of training data has label steering angle 0, which might cause issues. So I tried to filter out 99% of them using ```random()``` .

![vis-image1]

2. I use images from all three cameras, and an "angle correction" of 0.25 .

### * Appropriate training data

1. At first I tried to generate training data in simulator training mode, but my model didn't do well.
In order to reduce changing factors I use the "default udacity dataset" instead.

## Architecture and Training Documentation

### * Is the creation of the training dataset and training process documented?
As previously mentioned, I applied augmentation to the training dataset, here are examples of these augmentation:

* There are images taken from 3 camera of different angles: center, left, right, I use all 3 of them.

  Image from center camera:<br>
  ![vis-aug-1]

  Image from left camera:<br>
  ![vis-aug-2]

  Image from right camera:<br>
  ![vis-aug-3]

* Cropping the images.<br>
  ![vis-aug-4]
* Resizing the images.<br>
  ![vis-aug-5]
* Flippping 50 percent of the images horizontally.<br>
  ![vis-aug-6]
* Normalizing the images:

  Resized image after normalization:<br>
  ![vis-aug-7]
  
  Original 'center' image after normalization:<br>
  ![vis-aug-8]
