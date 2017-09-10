import csv
import cv2
import numpy as np
from random import randint

def add_driving_data(path, images, measurements):

    lines = []
    # 0.1 = 0.043 radians, according to
    # https://hoganengineering.wixsite.com/randomforest/ \
    #         single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles
    correction = 0.1; # parameter to tune

    with open(path + '/driving_log.csv') as csvfile:
        reader= csv.reader(csvfile);
        for line in reader:
            lines.append(line)
    for line in lines:
        ##print(line);
        filenames = []
        filenames.append(line[0].split('/')[-1])
        filenames.append(line[1].split('/')[-1])
        filenames.append(line[2].split('/')[-1])
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        ##if line[4] != 0: # steering info unhelpful if throttle is 0
        ##if float(line[4]) < 0.001 or (steering_center < 0.01 and randint(0,100) < 10):
        if True:
            # add image and angle
            image_center = cv2.imread(path + '/IMG/' + filenames[0])
            if randint(0,100) > 50:
                images.append(image_center)
                measurements.append(steering_center)
            else:
                # flip the center image, reverse the angle
                images.append(np.fliplr(image_center))
                measurements.append(-steering_center)
            image_left = cv2.imread(path + '/IMG/' + filenames[1])
            images.append(image_left)
            measurements.append(steering_left)
            image_right = cv2.imread(path + '/IMG/' + filenames[2])
            images.append(image_right)
            measurements.append(steering_right)
    return (images, measurements)

images, measurements = add_driving_data('./data/lake-dataCCW', [], [])
images, measurements = add_driving_data('./data/lake-dataCW', images, measurements)
images, measurements = add_driving_data('./data/jungle-dataCCW', images, measurements)
images, measurements = add_driving_data('./data/lake-dirtroad-turn-repetitive', images, measurements)


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# LeNET
#model = Sequential()
 ## model.add(Flatten(input_shape=(160,320,3))) # orig, most simple
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # normalize and mean center
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

# NVIDIA
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalize and mean center
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.20, shuffle=True, nb_epoch=5)

model.save('model.h5')
