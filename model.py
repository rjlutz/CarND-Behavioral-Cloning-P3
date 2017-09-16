import csv
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys

def add_driving_data(path, total_list):

    straight_list = []
    left_list = []
    right_list= []

    lines = []
    with open(path + '/driving_log.csv') as csvfile:
        reader= csv.reader(csvfile);
        for line in reader:
            lines.append(line)

    for line in lines:
        filenames = []
        for i in range(3):
            filenames.append(line[i].split('/')[-1])
        steering_center = float(line[3])

        # add images and angle
        paths = []
        paths.append(path + '/IMG/' + filenames[0])
        paths.append(path + '/IMG/' + filenames[1])
        paths.append(path + '/IMG/' + filenames[2])

        ## Straightest steering - undersample by 20%
        if (abs(steering_center) < 0.02 and random.random() <= 0.80):
            straight_list.append([paths[0], paths[1], paths[2], steering_center])
        # right turning, duplicate and tweak a little
        if (steering_center >0.2 and steering_center <=0.5):
            tweak = np.random.uniform(-1,1)/100.0
            right_list.append([paths[0], paths[1], paths[2], steering_center])
            right_list.append([paths[0], paths[1], paths[2], steering_center * (1.0 + tweak)])
        # left turning, dup and tweak
        elif (steering_center >= -0.5 and steering_center < -0.2):
            tweak = np.random.uniform(-1,1)/100.0
            left_list.append([paths[0], paths[1], paths[2], steering_center])
            left_list.append([paths[0], paths[1], paths[2], steering_center * (1.0 + tweak)])
        else: ## values between +/- (.02, .2) OR greater than +/- 0.5
            straight_list.append([paths[0], paths[1], paths[2], steering_center]);

    total_list += straight_list + left_list + right_list
    print ("Straight/Left/Right/Running Total = {} {} {} {}".format( \
               len(straight_list), len(left_list), len(right_list), len(total_list)))
    return total_list

fnames = []
fnames.extend(['./data/lake-dataCCW'])
fnames.extend(['./data/data-CCW-AJL'])
fnames.extend(['./data/lake-dataCW'])
##fnames.append(['./data/jungle-dataCCW'])([])

observations = []
for f in fnames:
    observations = add_driving_data(f, observations)

## add and amplify dirt road and curve data, adding some minor jitter each time
focused = []
focused.extend(['./data/lake-dirtroad-turn-repetitive'])
focused.extend(['./data/lake-firstturn-repetitive'])
focused.extend(['./data/lake-thirdturn-repetitive''])
focused.extend(['./data/data-corrections'])
focused.extend(['./data/bridge-repetitive'])
for f in focused:
    times = 4
    observations = add_driving_data(dset, observations)

    # turns_file = dset + '/' +  'driving_log.csv'
    #
    # augmented_data = pd.read_csv(turns_file, header=0)
    # augmented_data.columns = ["c_image", "l_image", "r_image", "steering", "throttle", "brake", "speed"]
    # for i in range(times):
    #     turns_list = []
    #     for j in range(len(augmented_data)):
    #         paths = []
    #         paths.append(dset + '/IMG/' + augmented_data["c_image"][j].split('/')[-1]);
    #         paths.append(dset + '/IMG/' + augmented_data["l_image"][j].split('/')[-1]);
    #         paths.append(dset + '/IMG/' + augmented_data["r_image"][j].split('/')[-1]);
    #         steer = augmented_data["steering"][j] * (1.0 + np.random.uniform(-1, 1) / 100.0)
    #         turns_list.append([paths[0], paths[1], paths[2], steer])
    #     observations += turns_list

# print ("Turns/Running Total = {} {}. Turns added {} times".format( \
#            len(turns_list), len(observations), times))

random.shuffle(observations)

from sklearn.model_selection import train_test_split
train_observations, validation_observations = train_test_split(observations, test_size=0.20)

print("Training/Validation/Total Observations {} {} {}".format(len(train_observations), \
      len(validation_observations), len(train_observations) + len(validation_observations)))

batch_size = 128

# Start with train generator shared in the class and add image augmentations
def train_generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while True: # Loop forever
        from sklearn.utils import shuffle
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # Read center, left and right images from a folder containing Udacity data and my data
            for batch_sample in batch_samples:
                center_image = cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(batch_sample[1]), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(batch_sample[2]), cv2.COLOR_BGR2RGB)

                steering_center = float(batch_sample[3])

                # Apply correction for left and right steering
                # create adjusted steering measurements for the side camera images
                # 0.1 = 0.043 radians, according to
                # https://hoganengineering.wixsite.com/randomforest/ \
                #         single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles
                correction = 0.2; # parameter to tune

                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # Randomly include either center, left or right image
                num = random.random()
                if num <= 0.33:
                    image = center_image
                    angle = steering_center
                elif num>0.33 and num<=0.66:
                    image = left_image
                    angle = steering_left
                else:
                    image = right_image
                    angle = steering_right

                images.append(image)
                angles.append(angle)

                # Randomly copy and flip selected images horizontally, with 75% probability
                if random.random() <= 0.75:
                    images.append(np.fliplr(image))
                    angles.append(-angle)

                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # Change to HSV
                hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.4, 1.2) # Convert back to RGB and append
                images.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
                angles.append(angle)

                # Randomly shear image with 80% probability
                if random.random() <= 0.80:
                    shear_range = 40
                    rows, cols, channels = image.shape
                    dx = np.random.randint(-shear_range, shear_range + 1)
                    random_point = [cols / 2 + dx, rows / 2]
                    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
                    pts2 = np.float32([[0, rows], [cols, rows], random_point])
                    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 10.0
                    M = cv2.getAffineTransform(pts1, pts2)
                    shear_image = cv2.warpAffine(center_image, M, (cols, rows), borderMode=1)
                    shear_angle = angle + dsteering
                    images.append(shear_image)
                    angles.append(shear_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def valid_generator(samples, batch_size=batch_size):
        num_samples = len(samples)
        while True:  # Loop forever so the generator never terminates
            from sklearn.utils import shuffle
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []
                #Validation generator: use center images only, no augmentation
                for batch_sample in batch_samples:
                    center_image = cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB)
                    images.append(center_image)
                    angles.append(float(batch_sample[3]))
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

train_generator = train_generator(train_observations, batch_size=batch_size)
validation_generator = valid_generator(validation_observations, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D

# NVIDIA
# model = Sequential()
# model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalize and mean center
# model.add(Cropping2D(cropping=((70,25),(0,0))))
#
# model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
# model.add(Dropout(0.5))
# model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
# model.add(Dropout(0.5))
# model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dropout(0.5))
#
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalize and mean center
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(8, 5, 5, border_mode='valid', activation='tanh')) # -> (66,316,8)
model.add(Dropout(0.5))
model.add(Convolution2D(16, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (31,156,16)
model.add(Dropout(0.5))
model.add(Convolution2D(20, 5, 5, border_mode='valid', activation='tanh', subsample=(2,2))) # -> (14,76,20)
model.add(Dropout(0.5))
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='tanh', subsample=(1,2))) # -> (10,36,24)
model.add(Dropout(0.5))
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='tanh', subsample=(1,2))) # -> (6,16,24)
model.add(Dropout(0.5))
model.add(Flatten()) # 6x16x24 -> 2304
from keras.regularizers import l2
model.add(Dense(30, activation='tanh', W_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(25, activation='tanh', W_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(20, activation='tanh', W_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh', W_regularizer=l2(0.01)))
model.compile(loss='mse', optimizer='adam')

model.summary()

nb_epoch = 14
samples_per_epoch = 20000
nb_val_samples = samples_per_epoch*0.20

history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, \
     validation_data=validation_generator, nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, \
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
plt.show()
plt.savefig("training_loss.png")
