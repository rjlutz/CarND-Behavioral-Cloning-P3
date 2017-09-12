import csv
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import sys;

def add_driving_data(path, images, measurements):

    lines = []
    # 0.1 = 0.043 radians, according to
    # https://hoganengineering.wixsite.com/randomforest/ \
    #         single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles
    correction = 0.2; # parameter to tune

    with open(path + '/driving_log.csv') as csvfile:
        reader= csv.reader(csvfile);
        for line in reader:
            lines.append(line)
    for line in lines:
        filenames = []
        for i in range(3):
            filenames.append(line[i].split('/')[-1])
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # steering info unhelpful if throttle is 0, DEAL WITH THIS??

        # add image and angle
        image_center = cv2.imread(path + '/IMG/' + filenames[0])

        ## Straightest steering - undersample by 20%
        if (abs(steering_center) < 0.02 and random.random() <= 0.80):
            if random.random() < 0.50:
                images.append(image_center)
                measurements.append(steering_center)
            else:
                images.append(np.fliplr(image_center)) # flip the center image, reverse angle
                measurements.append(-steering_center)
        # right turning, duplicate and tweak a little
        if (steering_center >0.2 and steering_center <=0.5):
            images.append(image_center)
            measurements.append(steering_center)
            tweak = np.random.uniform(-1,1)/100.0
            images.append(image_center)
            measurements.append(steering_center * (1.0 + tweak))
        # left turning, dup and tweak
        elif (steering_center >= -0.5 and steering_center < -0.2):
            images.append(image_center)
            measurements.append(steering_center)
            tweak = np.random.uniform(-1,1)/100.0
            images.append(image_center)
            measurements.append(steering_center * (1.0 + tweak))
        else: ## values between +/- (.02, .2) OR greater than +/- 0.5
            images.append(image_center)
            measurements.append(steering_center)

        # flip the lake-firstturn-repetitive image, reverse angle
        image_left = cv2.imread(path + '/IMG/' + filenames[1])
        images.append(image_left)
        measurements.append(steering_left)

        # flip the right image, reverse angle
        image_right = cv2.imread(path + '/IMG/' + filenames[2])
        images.append(image_right)
        measurements.append(steering_right)

    return (images, measurements)

fnames = []
fnames.extend(['./data/lake-dataCCW', './data/data-CCW-AJL']);
fnames.extend(['./data/data-corrections', './data/lake-dirtroad-turn-repetitive'])
##fnames.extend(['./data/lake-firstturn-repetitive', './data/lake-dataCW'])
##fnames.append(['./data/jungle-dataCCW'])

images = []
measurements = []
for f in fnames:
    images, measurements = add_driving_data(f, images, measurements)
    print("fname = {}, size = {}".format(f, len(images)))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# NVIDIA
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalize and mean center
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')

## backstop
##sys.exit(0);


history_object = model.fit(X_train, y_train, validation_split=0.20, shuffle=True, nb_epoch=5, \
    verbose=1)

model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
fig = plt.figure()
fig.savefig("training_loss.png")
