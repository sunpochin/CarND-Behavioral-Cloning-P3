import os
import csv
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers import Input, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

import cv2
import numpy as np
import sklearn

#traintag = 'train-4-few'
traintag = 'udacity-data/'
csvfilename = './' + traintag + '/driving_log.csv'
samples = []
with open(csvfilename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# create adjusted steering measurements for the side camera images
# TODO
correction = 0.25 # this is a parameter to tune
centerenum = 0
leftenum = 1
rightenum = 2
def read_image(name, angle, dir):
    image = cv2.imread(name)
    center_angle = angle
    if leftenum == dir: # left
        angle = center_angle + correction
    elif centerenum == dir: # center
        angle = center_angle
        # drop all 0 angle!
        # drop 75% angle 0.
        if 0 == angle:
            drop_prob = np.random.random()
            if drop_prob > 0.6:
                return None, 4
    elif rightenum == dir: # right
        angle = center_angle - correction
    else:
        pass

    if image.any() == None:
        print("Invalid image path:", center_name)
        return None, 4
    else:
        pass

    return image, angle

# decide whether to horizontally flip the image:
def decide_flip_image(image, angle):
    # https://discussions.udacity.com/t/using-generator-to-implement-random-augmentations/242185/7
    flip_prob = np.random.random()
    if flip_prob > 0.5:
    # flip the image and reverse the steering angle
        angle = -1 * angle
        image = cv2.flip(image, 1)

    return image, angle


bsize = 32
def generator(samples, batch_size = bsize):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # name on  ubuntu
                if batch_sample[3] == 'steering':
                    continue
                center_angle = float(batch_sample[3] )

                cameras_dir = [centerenum, leftenum, rightenum]
                for camera in cameras_dir:
                    name = './' + traintag + batch_sample[camera].strip().split('\\')[-1]
                    image, angle = read_image(name, center_angle, camera)
                    if (angle == 4):
    #                    print("None image! : ", camera)
                        continue
                    image, angle = decide_flip_image(image, angle)
                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
#            print('X_train.shape:', X_train.shape)
#            print('y_train.shape:', y_train.shape)
#            print('y_train:', y_train)
            some = sklearn.utils.shuffle(X_train, y_train)
#            print('some: ', some)
            yield some


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# trim image to only see section with road

# cropping
model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape = (row, col, ch) ) )
# The example above crops: 50 rows pixels from the top of the image 
#30 rows pixels from the bottom of the image
#0 columns of pixels from the left of the image 0 columns of pixels from the right of the image

# resize: https://discussions.udacity.com/t/keras-lambda-to-resize-seems-causing-the-problem/316247/3?u=sunpochin
def resize_img(input):
    # ktf must be declared here to 'be stored in the model' and 'let drive.py use it'
    new_width = 64
    new_height = 64
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (new_width, new_height))
# resize: https://discussions.udacity.com/t/keras-lambda-to-resize-seems-causing-the-problem/316247/3?u=sunpochin

model.add(Lambda(resize_img))
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape = (row, col, ch) ) )

# Preprocess incoming data, centered around zero with small standard deviation 
'''
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
#model.add(... finish defining the rest of your model architecture here ...)
'''
model.add( Conv2D(24, (5, 5), strides = (2, 2),
    padding = 'same', activation="relu") )
model.add( Conv2D(36, (5, 5), strides = (2, 2),
    padding = 'same', activation="relu") )

'''
model.add( Conv2D(48, (5, 5), strides = (2, 2),
    padding = 'same', activation="relu") )
model.add( Conv2D(64, (3, 3), strides = (1, 1), 
    padding = 'same', activation="relu") )
model.add( Conv2D(64, (3, 3), strides = (1, 1),
    padding = 'same', activation="relu") )
'''

model.add(Flatten() )
'''
model.add(Dense(1164) )
model.add(Dense(100) )
model.add(Dense(50) )
'''
#keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
model.add(Dense(20) )
model.add(Dropout(0.5) )
model.add(Dense(10) )
model.add(Dropout(0.5) )
model.add(Dense(10) )
model.add(Dense(1) )
from keras import optimizers
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer = adam)

'''
for i in range(32):    
    train = next(train_generator)
    print(train[0].shape)
'''
#print('train_generator', train_generator)

print('len(train_samples): ', len(train_samples) )
# use sample_rate and epoch for quicker test. 
# If I want to test something quick but rough, set a HIGHER sample_rate to reduce training
sample_rate = 32
epoch = 5
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('log.csv', append=True, separator=';')

loss_history = model.fit_generator(train_generator,
                    steps_per_epoch = len(train_samples) / sample_rate, 
                    validation_data = validation_generator,
                    validation_steps = len(validation_samples) / sample_rate, epochs = epoch,
                    callbacks=[csv_logger])

model.save('model.h5')

#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')
#https://stackoverflow.com/questions/38445982/how-to-log-keras-loss-output-to-a-file
import numpy as np
#numpy_loss_history = np.array(loss_history)
#np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

# 
# http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# list all data in history
print(loss_history.history.keys())
# summarize history for accuracy
plt.plot(loss_history.history['acc'])
plt.plot(loss_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss_history.history['loss'])
plt.plot(loss_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
