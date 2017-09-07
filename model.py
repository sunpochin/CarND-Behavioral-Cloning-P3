import os, csv, cv2, sklearn
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Flatten, Dense, Lambda, Dropout
# from keras.layers import Input, Cropping2D
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Model
# from keras.callbacks import CSVLogger
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# use sample_rate and epoch for quicker test.
# If I want to test something quick but rough, set a HIGHER sample_rate to reduce training
bsize = 32
epoch = 5
down_sample_rate = 1
#down_sample_rate = 1 * bsize

#traintag = 'train-4-few'
traintag = 'udacity-data/'

def GetSampleFullname():
    csvfilename = './' + traintag + '/driving_log.csv'
    samples = []
    with open(csvfilename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

samples = GetSampleFullname()
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create adjusted steering measurements for the side camera images
# TODO
correction = 0.25 # this is a parameter to tune

from enum import Enum
class columnIdx(Enum):
    center = 0
    left = 1
    right = 2
    steering = 3


#
def read_image(name, steering_angle, carmeradir):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # TODO: I should here save some images for the writeup
    if columnIdx.left == carmeradir: # left
        angle = steering_angle + correction
    elif columnIdx.center == carmeradir: # center
        angle = steering_angle
        # drop all 0 angle!
        # only keeping 10% angle 0.
        # https://discussions.udacity.com/t/vehicle-drives-in-circles-in-autonomous-mode-what-could-be-going-wrong/283222/3?u=sunpochin
        if 0 == angle:
            drop_prob = np.random.random()
            # this is a parameter to tune
            if drop_prob > 0.01:
                return None, None
    elif columnIdx.right == carmeradir: # right
        angle = steering_angle - correction
    else:
        pass

    if image.any() == None:
        print("Invalid image path:", center_name)
        return None, None
    else:
        pass

    return image, angle

# flipt 50 percent of the images horizontally:
def flip_50_percent_image(image, angle):
    # https://discussions.udacity.com/t/using-generator-to-implement-random-augmentations/242185/7
    flip_prob = np.random.random()
    if flip_prob > 0.5:
    # flip the image and reverse the steering angle
        angle = -1 * angle
        image = cv2.flip(image, 1)
    return image, angle

# used to fit_generator in batch, avoid out of memory.
def generator(samples, batch_size = bsize):
    num_samples = len(samples)
    print('num_samples: ', num_samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            # print('offset: ', offset)
            images = []
            angles = []
            for batch_sample in batch_samples:
                # to skip the first row: column name.
                if batch_sample[columnIdx.steering.value] == 'steering':
                    continue
                steering_angle = float(batch_sample[columnIdx.steering.value] )
                cameras_dir = [columnIdx.center, columnIdx.left, columnIdx.right]
                for camera in cameras_dir:
                    name = './' + traintag + batch_sample[camera.value].strip().split('\\')[-1]
                    image, angle = read_image(name, steering_angle, camera)
                    if (None == image):
#                        print("None image! : ", camera)
                        continue
                    image, angle = flip_50_percent_image(image, angle)
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


# resize: https://discussions.udacity.com/t/keras-lambda-to-resize-seems-causing-the-problem/316247/3?u=sunpochin
def resize_img(input):
    # ktf must be declared here to 'be stored in the model' and 'let drive.py use it'
    new_width = 64
    new_height = 64
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (new_width, new_height))
# resize: https://discussions.udacity.com/t/keras-lambda-to-resize-seems-causing-the-problem/316247/3?u=sunpochin


def getmodel():
    from keras.models import Sequential
    from keras.layers.core import Flatten, Dense, Lambda, Dropout
    from keras.layers import Input, Cropping2D
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Model

    ch, row, col = 3, 160, 320  # Trimmed image format

    model = Sequential()
    # cropping image to only see section with road
    model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape = (row, col, ch) ) )
    # The example above crops: 50 rows pixels from the top of the image
    #30 rows pixels from the bottom of the image
    #0 columns of pixels from the left of the image 0 columns of pixels from the right of the image
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
    model.add( Conv2D(48, (5, 5), strides = (2, 2),
        padding = 'same', activation="relu") )

    model.add( Conv2D(64, (3, 3), strides = (1, 1),
        padding = 'same', activation="relu") )
    model.add( Conv2D(64, (3, 3), strides = (1, 1),
        padding = 'same', activation="relu") )

    model.add(Flatten() )

    model.add(Dense(1164) )
    model.add(Dropout(0.2) )
    model.add(Dense(100) )
    model.add(Dropout(0.2) )
    model.add(Dense(50) )
    model.add(Dropout(0.2) )
    model.add(Dense(10) )
    model.add(Dense(1) )

    from keras import optimizers
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='mse', optimizer = adam)
    return model

'''
for i in range(32):
    train = next(train_generator)
    print(train[0].shape)
'''

def train_model(model):
    from keras.callbacks import CSVLogger
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    print('len(train_samples): ', len(train_samples) )
    csv_logger = CSVLogger('log.csv', append=True, separator=';')


    '''
                 ReduceLROnPlateau(monitor='val_dice_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                EarlyStopping(monitor='val_loss',
                               patience=8,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
    '''
    # the best_weights actually won't work because in drive.py uses load_model().
    callbacks = [
                ModelCheckpoint(monitor='val_loss',
                                 filepath='weights/best_weights.hdf5',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max'),
                TensorBoard(log_dir='logs'),
                CSVLogger('log.csv', append=True, separator=';') ]


    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size = bsize)
    validation_generator = generator(validation_samples, batch_size = bsize)
    loss_history = model.fit_generator(train_generator,
                        steps_per_epoch = len(train_samples) / down_sample_rate,
                        validation_data = validation_generator,
                        validation_steps = len(validation_samples) / down_sample_rate,
                        epochs = epoch,
                        callbacks=callbacks)

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
    import matplotlib.pyplot as plt
    # summarize history for loss
    plt.plot(loss_history.history['loss'])
    plt.plot(loss_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    model = getmodel()
    train_model(model)
    pass
