import os
import csv
from keras.models import Sequential
from keras.layers.core  import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Cropping2D
import cv2
import numpy as np
import sklearn

traintag = 'train-4-few'
csvfilename = './' + traintag + '/driving_log.csv'
samples = []
with open(csvfilename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


bsize = 32
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # name on  ubuntu
                name = './' + traintag + '/IMG/' + batch_sample[0].split('\\')[-1]
#                name = './train-4/IMG/' + batch_sample[0].split('\\')[-1]
#                print(name)
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                if center_image == None:
                    print("Invalid image path:", name)
                else:
                    angle = float(center_angle)
                    # avoid "going straight" bias. So I totally dropped all angle 0 data!
                    if 0.0 != angle:
                        images.append(center_image)
                        angles.append(angle)
                    else:
                        #print("skipped angle 0!")
                        pass


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
#            print('X_train.shape:', X_train.shape)
#            print('y_train.shape:', y_train.shape)
#            print('y_train:', y_train)
            some = sklearn.utils.shuffle(X_train, y_train)
#            print('some: ', some)
            yield some
#            some = (X_train, y_train)
#            print('type of some:', type(some) )
#            yield some
#            yield X_train, y_train


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

#ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# cropping
model.add(Cropping2D(cropping=((80, 0), (80,80)), input_shape=(160,320,3)))
'''
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# The example above crops:
50 rows pixels from the top of the image
20 rows pixels from the bottom of the image
0 columns of pixels from the left of the image
0 columns of pixels from the right of the image
'''

# Preprocess incoming data, centered around zero with small standard deviation 
'''
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
#model.add(... finish defining the rest of your model architecture here ...)
'''
#model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape = (160, 320, 3) ) )
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape = (160, 320, 3) ) )

model.add( Conv2D(32, (5, 5), 
    padding = 'same',
    activation="relu") )
model.add( Conv2D(32, (5, 5), 
    padding = 'same',
    activation="relu") )
model.add(Flatten() )
model.add(Dense(16) )
model.add(Dense(8) )
model.add(Dense(1) )

model.compile(loss='mse', optimizer='adam')

'''
for i in range(32):    
    train = next(train_generator)
    print(train[0].shape)
'''
#print('train_generator', train_generator)

print('len(train_samples): ', len(train_samples) )

'''
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
    validation_data = validation_generator, 
    nb_val_samples = len(validation_samples), epochs=3)
    '''


model.fit_generator(train_generator,
                    steps_per_epoch = len(train_samples) , 
                    validation_data = validation_generator,
                    validation_steps = len(validation_samples) , epochs = 5)

model.save('model.h5')
