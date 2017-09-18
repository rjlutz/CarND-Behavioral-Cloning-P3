import csv
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg') ## needed for lights-out plotting
import matplotlib.pyplot as plt

batch_size = 128

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
        # inspiration from https://github.com/priya-dwivedi/CarND/tree/master/CarND-Behavior-Cloning-P3
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
fnames.extend(['./data/data-provided'])          ## provided with the project

## add and amplify dirt road and curve data, adding some minor jitter each time
focused = []
focused.extend(['./data/lake-dirtroad-turn-repetitive'])
focused.extend(['./data/lake-firstturn-repetitive'])
focused.extend(['./data/lake-thirdturn-repetitive'])
focused.extend(['./data/data-corrections'])
focused.extend(['./data/bridge-repetitive'])

observations = []
for f in fnames:
    observations = add_driving_data(f, observations)
for f in focused:
    times = 4
    for i in range(times):
        observations = add_driving_data(f, observations)

random.shuffle(observations)

from sklearn.model_selection import train_test_split
train_observations, validation_observations = train_test_split(observations, test_size=0.20)

print("Training/Validation/Total Observations {} {} {}".format(len(train_observations), \
      len(validation_observations), len(train_observations) + len(validation_observations)))

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

                # diagnostic images
                skip = 100 ## interval, skip this many batches before printing
                if offset % skip == 0 and i == 0:
                    plotImageTrio(center_image, left_image, right_image,
                    './diagnostic_images/training_generator_LRC_{0:.0f}.png'.format(offset),
                    'Periodic selection (index: {}) from training generator.\n Steering L/C/R = {:.2f} {:.2f} {:.2f}'
                    .format(offset, steering_left, steering_center, steering_right))

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

                # Randomly shadow image and add to images with 80% probability
                ## inspiration from https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py
                x1, y1 = image.shape[1] * np.random.rand(), 0
                x2, y2 = image.shape[1] * np.random.rand(), image.shape[0]
                xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]] # xm, ym gives all the locations of the image
                mask = np.zeros_like(image[:, :, 1])
                mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

                coin = mask == np.random.randint(2) # choose which side should have shadow and adjust saturation
                s_ratio = np.random.uniform(low=0.2, high=0.5)
                hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS) # adjust Saturation in HLS(Hue, Light, Saturation)
                hls[:, :, 1][coin] = hls[:, :, 1][coin] * s_ratio
                shadow_image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
                if random.random() <= 0.80:
                    images.append(shadow_image)
                    angles.append(angle)

                # diagnostic images
                if offset % skip == 0 and i == 0:
                    plotImageTrio(flipped_image, bright_image, shadow_image,
                               './diagnostic_images/training_generator_augmented-{0:.0f}.png'.format(offset),
                               'Periodic selection (index: {}) from training generator.\n Augmented Flipped/Brightened/Shadow'
                               .format(offset))

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
                    images.append(cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB))
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

# NVIDIA, modified, inspiration from https://github.com/naokishibuya/car-behavioral-cloning
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalize and mean center
model.add(Cropping2D(cropping=((60,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='elu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))

model.summary()

history_object = model.fit_generator(train_generator, samples_per_epoch=20224, \
     validation_data=validation_generator, nb_val_samples=20224*0.20, \
     nb_epoch=10, verbose=1)

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
