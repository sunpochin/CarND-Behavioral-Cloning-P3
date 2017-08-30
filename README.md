# Behaviorial Cloning Project

Overview
---
This is my udacity self driving car project, which tries to train a model to predict which "steering angle" to use according to different situations.

A recording of the simulation could be found here:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4reOzBoT5M/0.jpg)](https://www.youtube.com/watch?v=O4reOzBoT5M)


ConvNet Architecture
---
I use the network architecture nvidia used for their self driving car. [Link here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

Implementation details and parameters
---
1. I use the "default udacity dataset" to reduce changing factors.
2. From some basic EDA I learned a great portion of training data has label steering angle 0, which might cause issues. So I tried to filter out 99% of them using ```random()``` .
3. I use images from all three cameras, and a angle correction of 0.25 .


