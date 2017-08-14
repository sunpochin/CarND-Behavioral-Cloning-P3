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
	for i in range(3):
		source_path = line[0]



		image = cv2.imread(source_path)

		images.append(image)


		measurement = float(line[3])
		measurements.append(measurement)



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
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 7)

model.save('model.h5')












