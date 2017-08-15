import csv
import cv2

lines = []
with open('training-3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]

	image = cv2.imread(source_path)

	images.append(image)


	measurement = float(line[3])
	measurements.append(measurement)


# data augmentation
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement) 
	augmented_images.append(cv2.flip(image, 1) )
	augmented_measurements.append(measurement * -1.0)
images = augmented_images
measurements = augmented_measurements


import numpy as np
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core  import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
# pre-processing
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape = (160, 320, 3) ) )
# LeNet
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten() )
model.add(Dense(120) )
model.add(Dense(84) )
model.add(Dense(1) )

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5, verbose = 2)

model.save('model.h5')

'''
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

'''