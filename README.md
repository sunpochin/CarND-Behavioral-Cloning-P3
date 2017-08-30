# Behaviorial Cloning Project

Overview
---
This is my udacity self driving car project, which tries to train a model to predict which "steering angle" to use according to different situations.

A recording of the simulation could be found here:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4reOzBoT5M/0.jpg)](https://www.youtube.com/watch?v=O4reOzBoT5M)


ConvNet Architecture
---
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
 



Implementation details and parameters
---
1. I use the "default udacity dataset" to reduce changing factors.
2. From some basic EDA I learned a great portion of training data has label steering angle 0, which might cause issues. So I tried to filter out 99% of them using ```random()``` .
3. I use images from all three cameras, and a angle correction of 0.25 .


