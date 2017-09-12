import csv
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys;

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

        # steering info unhelpful if throttle is 0, DEAL WITH THIS??

        # add images and angle
        paths = [];
        paths.append(path + '/IMG/' + filenames[0]);
        paths.append(path + '/IMG/' + filenames[1]);
        paths.append(path + '/IMG/' + filenames[2]);

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

print("Training/Validation/Total Observations {} {} {}".format(len(train_observations), \
      len(validation_observations), len(train_observations) + len(validation_observations)))

batch_size = 128;

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
                center_name = batch_sample[0]
                center_image = cv2.imread(center_name)
                ##center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_name = batch_sample[1]
                left_image = cv2.imread(left_name)
                ##left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_name = batch_sample[2]
                right_image = cv2.imread(right_name)
                ##right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                steering_center = float(batch_sample[3])

                # Apply correction for left and right steering
                correction = 0.2; # parameter to tune

                # create adjusted steering measurements for the side camera images
                # 0.1 = 0.043 radians, according to
                # https://hoganengineering.wixsite.com/randomforest/ \
                #         single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles
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
                    ##center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    images.append(cv2.imread(batch_sample[0]))
                    angles.append(float(batch_sample[3]))

                X_train = np.array(images)
                y_train = np.array(angles)

                yield shuffle(X_train, y_train)


train_generator = train_generator(train_observations, batch_size=batch_size)
validation_generator = valid_generator(validation_observations, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam

# Function to resize image to 64x64
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image,[40,60])


#Params
row, col, ch = 160, 320, 3
nb_classes = 1

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))
# Crop pixels from top and bottom of image
model.add(Cropping2D(cropping=((60, 20), (0, 0))))

# Resise data within the neural network
model.add(Lambda(resize_image))
# Normalize data
model.add(Lambda(lambda x: (x / 127.5 - 1.)))

# First convolution layer so the model can automatically figure out the best color space for the hypothesis
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))

# CNN model

model.add(Convolution2D(32, 3,3 ,border_mode='same', subsample=(2,2), name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1), name='pool1'))

model.add(Convolution2D(64, 3,3 ,border_mode='same',subsample=(2,2), name='conv2'))
model.add(Activation('relu',name='relu2'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool2'))

model.add(Convolution2D(128, 3,3,border_mode='same',subsample=(1,1), name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2), name='pool3'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, name='dense2'))

model.add(Dense(1,name='output'))

model.compile(optimizer=Adam(lr= 0.0001), loss="mse")

## backstop
##sys.exit(0);

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
