import csv
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys;

def add_driving_data(path, total_list):

    lines = []
    # 0.1 = 0.043 radians, according to
    # https://hoganengineering.wixsite.com/randomforest/ \
    #         single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles

    straight_list = []
    left_list = []
    right_list= []

    with open(path + '/driving_log.csv') as csvfile:
        reader= csv.reader(csvfile);
        for line in reader:
            lines.append(line)

    for line in lines:
        filenames = []
        for i in range(3):
            filenames.append(line[i].split('/')[-1])
        steering_center = float(line[3])

        # steering info unhelpful if throttle is 0, DEAL WITH THIS??

        # add images and angle
        paths = [];
        paths.append(path + '/IMG/' + filenames[0]);
        paths.append(path + '/IMG/' + filenames[1]);
        paths.append(path + '/IMG/' + filenames[2]);
        ##image_center = cv2.imread(paths[0])
        ##image_left = cv2.imread(paths[1])
        ##image_right = cv2.imread(paths[2])

        ## Straightest steering - undersample by 20%
        if (abs(steering_center) < 0.02 and random.random() <= 0.80):
            straight_list.append([path[0], path[1], path[2], steering_center])
        # right turning, duplicate and tweak a little
        if (steering_center >0.2 and steering_center <=0.5):
            tweak = np.random.uniform(-1,1)/100.0
            right_list.append([path[0], path[1], path[2], steering_center])
            right_list.append([path[0], path[1], path[2], steering_center * (1.0 + tweak)])
        # left turning, dup and tweak
        elif (steering_center >= -0.5 and steering_center < -0.2):
            tweak = np.random.uniform(-1,1)/100.0
            left_list.append([path[0], path[1], path[2], steering_center])
            left_list.append([path[0], path[1], path[2], steering_center * (1.0 + tweak)])
        else: ## values between +/- (.02, .2) OR greater than +/- 0.5
            straight_list.append([path[0], path[1], path[2], steering_center]);

    total_list += straight_list + left_list + right_list
    print ("Straight/Left/Right/Running Total = {} {} {} {}".format( \
               len(straight_list), len(left_list), len(right_list), len(total_list)))
    return total_list

fnames = []
fnames.extend(['./data/lake-dataCCW', './data/data-CCW-AJL']);
fnames.extend(['./data/data-corrections'])
for i in range(4): ## amplify dirt road data, adding some minor jitter each time
    fnames.extend(['./data/lake-dirtroad-turn-repetitive'])
##fnames.extend(['./data/lake-firstturn-repetitive', './data/lake-dataCW'])
##fnames.append(['./data/jungle-dataCCW'])

observations = []
for f in fnames:
    observations = add_driving_data(f, observations)

random.shuffle(observations)

from sklearn.model_selection import train_test_split
train_observations, validation_observations = train_test_split(observations, test_size=0.20)

print("Training/Validation Observations {} {}".format(len(train_observations), \
      len(validation_observations)))

batch_size = 128;

# Start with train generator shared in the class and add image augmentations
def train_generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        from sklearn.utils import shuffle
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # Read center, left and right images from a folder containing Udacity data and my data
            for batch_sample in batch_samples:
                center_name = batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                ##center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_name = batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                ##left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_name = batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                ##right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                steering_center = float(batch_sample[3])

                # Apply correction for left and right steering
                correction = 0.2; # parameter to tune

                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # Randomly include either center, left or right image
                num = random.random()
                if num <= 0.33:
                    image = center_image
                    angle = steering_center
                elif num>0.33 and num<=0.66:
                    image = left_image
                    angle = left_angle
                else:
                    image = right_image
                    angle = right_angle

                images.append(image)
                angles.append(angle)

                # Randomly copy and flip selected images horizontally, with 75% probability
                if random.random() >0.25:
                    ##flip_image = np.fliplr(image)
                    ##flip_angle = -1*angle
                    images.append(np.fliplr(image))
                    angles.append(-angle)

                # # Augment with images of different brightness
                # # Randomly select a percent change
                # change_pct = random.uniform(0.4, 1.2)
                #
                # # Change to HSV to change the brightness V
                # hsv = cv2.cvtColor(select_image, cv2.COLOR_RGB2HSV)
                #
                # hsv[:, :, 2] = hsv[:, :, 2] * change_pct
                # # Convert back to RGB and append
                #
                # bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                # images.append(bright_img)
                # angles.append(select_angle)

                ## Randomly shear image with 80% probability
                # shear_prob = random.random()
                # if shear_prob >=0.20:
                #     shear_range = 40
                #     rows, cols, ch = select_image.shape
                #     dx = np.random.randint(-shear_range, shear_range + 1)
                #     #    print('dx',dx)
                #     random_point = [cols / 2 + dx, rows / 2]
                #     pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
                #     pts2 = np.float32([[0, rows], [cols, rows], random_point])
                #     dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 10.0
                #     M = cv2.getAffineTransform(pts1, pts2)
                #     shear_image = cv2.warpAffine(center_image, M, (cols, rows), borderMode=1)
                #     shear_angle = select_angle + dsteering
                #     images.append(shear_image)
                #     angles.append(shear_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

def valid_generator(samples, batch_size=batch_size):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            from sklearn.utils import shuffle
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []

                #Validation generator only has center images and no augmentations
                for batch_sample in batch_samples:
                    center_name = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data2/IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(center_name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                    center_angle = float(batch_sample[3])

                    images.append(center_image)
                    angles.append(center_angle)

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

                #center images only
                for batch_sample in batch_samples:
                    ##center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    images.append(cv2.imread(center_name))
                    angles.append(float(batch_sample[3]))

                X_train = np.array(images)
                y_train = np.array(angles)

                yield shuffle(X_train, y_train)

train_generator = train_generator(train_observations, batch_size=batch_size)
validation_generator = valid_generator(validation_observations, batch_size=batch_size)

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
sys.exit(0);

##history_object = model.fit(X_train, y_train, validation_split=0.20, shuffle=True, nb_epoch=5, \
##    verbose=1)

history_object = model.fit_generator(train_generator, samples_per_epoch=20000, \
     validation_data=validation_generator, nb_val_samples=2000, nb_epoch=5, verbose=1)


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
