import csv
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

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

        if float(line[4]) < 0.25: ## skip if little or no throttle
            continue

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
fnames.extend(['./data/lake-dataCCW'])
##fnames.append(['./data/jungle-dataCCW'])([])

observations = []
for f in fnames:
    observations = add_driving_data(f, observations)

## add and amplify dirt road and curve data, adding some minor jitter each time
focused = []
focused.extend(['./data/lake-dirtroad-turn-repetitive'])
focused.extend(['./data/lake-firstturn-repetitive'])
focused.extend(['./data/lake-thirdturn-repetitive'])
focused.extend(['./data/data-corrections'])
focused.extend(['./data/bridge-repetitive'])
for f in focused:
    times = 4
    for i in range(times):
        observations = add_driving_data(f, observations)

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

def cropScaleImage(image):
    # shape = image.shape
    # image = image[math.floor(60):shape[0]-20, 0:shape[1]]  # note: numpy arrays are (row, col)!
    # image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    return image

def plotImageTrio(c, l, r, name, title):
    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(l);
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(c);
    plt.axis('off')
    plt.title(title)
    plt.subplot(1,3,3)
    plt.imshow(r);
    plt.axis('off');
    plt.savefig(name);
    plt.close(fig)

def plotRandomImage(dset, name, title):
    ind_num = random.randint(0,len(train_observations))
    observation = dset[ind_num]
    image_c = cv2.cvtColor(cv2.imread(observation[0]), cv2.COLOR_BGR2RGB)
    image_l = cv2.cvtColor(cv2.imread(observation[1]), cv2.COLOR_BGR2RGB)
    image_r = cv2.cvtColor(cv2.imread(observation[2]), cv2.COLOR_BGR2RGB)
    plotImageTrio(image_c, image_l, image_r, name, title)


# visual validation of data set
plotRandomImage(train_observations, 'example-camera-images.png', 'L/R/C')

batch_size = 256
new_size_row = 64
new_size_col = 64

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
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                center_image = cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(batch_sample[1]), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(batch_sample[2]), cv2.COLOR_BGR2RGB)
                center_image = cropScaleImage(center_image)
                right_image = cropScaleImage(right_image)
                left_image = cropScaleImage(left_image)

                steering_center = float(batch_sample[3])

                # Apply correction for left and right steering
                # create adjusted steering measurements for the side camera images
                # 0.1 = 0.043 radians, according to
                # https://hoganengineering.wixsite.com/randomforest/ \
                #         single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles
                correction = 0.20; # parameter to tune

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

                skip = 100 ## interval
                if offset % skip == 0 and i == 0:
                    plotImageTrio(center_image, left_image, right_image,
                    './diagnostic_images/training_generator_LRC_{0:.0f}.png'.format(offset * batch_size),
                    'Periodic selection (index: {}) from training generator.\n Steering L/C/R = {:.2f} {:.2f} {:.2f}'
                    .format(offset * batch_size, steering_left, steering_center, steering_right))

                # Randomly copy and flip selected images horizontally, with 90% probability
                flipped_image = np.fliplr(image)
                if random.random() <= 0.90:
                    images.append(flipped_image)
                    angles.append(-angle)

                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # Change to HSV
                hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.4, 1.2) # Convert back to RGB and append
                bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                images.append(bright_image)
                angles.append(angle)

                # Randomly shadow image with 80% probability
                ## inspiration from https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py
                w = 320
                h = 160
                x1, y1 = w * np.random.rand(), 0
                x2, y2 = w * np.random.rand(), h
                xm, ym = np.mgrid[0:h, 0:w] # xm, ym gives all the locations of the image
                mask = np.zeros_like(image[:, :, 1])
                mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

                # choose which side should have shadow and adjust saturation
                cond = mask == np.random.randint(2)
                s_ratio = np.random.uniform(low=0.2, high=0.5)

                # adjust Saturation in HLS(Hue, Light, Saturation)
                hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
                shadow_image =  cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
                # shear_range = 40
                # rows, cols, channels = image.shape
                # dx = np.random.randint(-shear_range, shear_range + 1)
                # random_point = [cols / 2 + dx, rows / 2]
                # pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
                # pts2 = np.float32([[0, rows], [cols, rows], random_point])
                # dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 10.0
                # M = cv2.getAffineTransform(pts1, pts2)
                # shear_image = cv2.warpAffine(center_image, M, (cols, rows), borderMode=1)
                # shear_angle = angle + dsteering
                if random.random() <= 0.80:
                    images.append(shadow_image)
                    angles.append(angle)

                if offset % skip == 0 and i == 0:
                    plotImageTrio(flipped_image, bright_image, shadow_image,
                               './diagnostic_images/training_generator_augmented-{0:.0f}.png'.format(offset * batch_size),
                               'Periodic selection (index: {}) from training generator.\n Augmented Flipped/Brightened/Shadow'
                               .format(offset * batch_size))

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
                    center_image = cropScaleImage(center_image)
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
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.optimizers import Adam

# # Function to resize image
# def resize_image(image):
#     import tensorflow as tf
#     return tf.image.resize_images(image,[new_size_row,new_size_col])
#
# ## try this model
# filter_size = 3
# pool_size = (2,2)
#
# model = Sequential() ## Vivek
# model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(new_size_row, new_size_col, 3))) # normalize and mean center
# # Resise data within the neural network
# model.add(Convolution2D(3,1,1, border_mode='valid',name='conv0', init='he_normal'))
# model.add(ELU())
# model.add(Convolution2D(32,filter_size,filter_size,border_mode='valid',name='conv1', init='he_normal'))
# model.add(ELU())
# model.add(Convolution2D(32,filter_size,filter_size,border_mode='valid',name='conv2', init='he_normal'))
# model.add(ELU())
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.5))
# model.add(Convolution2D(64,filter_size,filter_size,border_mode='valid',name='conv3', init='he_normal'))
# model.add(ELU())
# model.add(Convolution2D(64,filter_size,filter_size,border_mode='valid',name='conv4', init='he_normal'))
# model.add(ELU())
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.5))
# model.add(Convolution2D(128,filter_size,filter_size,border_mode='valid',name='conv5', init='he_normal'))
# model.add(ELU())
# model.add(Convolution2D(128,filter_size,filter_size,border_mode='valid',name='conv6', init='he_normal'))
# model.add(ELU())
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
#
# model.add(Dense(512,name='hidden1', init='he_normal'))
# model.add(ELU())
# model.add(Dropout(0.5))
# model.add(Dense(64,name='hidden2', init='he_normal'))
# model.add(ELU())
# model.add(Dropout(0.5))
# model.add(Dense(16,name='hidden3',init='he_normal'))
# model.add(ELU())
# model.add(Dropout(0.5))
# model.add(Dense(1, name='output', init='he_normal'))
#
# i##--##adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer='adam', loss='mse')


# NVIDIA
# model = Sequential()
#
# model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalize and mean center
# model.add(Cropping2D(cropping=((70,25),(0,0))))
#
# # Resise data within the neural network
# model.add(Lambda(resize_image))
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
# model.compile(loss='mse', optimizer='adam')

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3))) # normalize and mean center
model.add(Cropping2D(cropping=((60,25),(0,0))))
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

nb_epoch = 15
samples_per_epoch = 30720
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
