import csv
import cv2
import sklearn
from sklearn.utils import shuffle
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    lines.append(line)

import random
from random import randint
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32, is_training=False):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        if is_training == True:
          cam_id = randint(0,2)
        else :
          cam_id = 0
        name = './data/IMG/'+batch_sample[cam_id].split('/')[-1]
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        angle = float(batch_sample[3])
        if cam_id == 1:
          angle += 0.2
        elif cam_id == 2:
          angle -= 0.2
        if is_training == True and random.random() > 0.5:
          image = cv2.flip(image,1)
          angle = -angle
        images.append(image)
        angles.append(angle)

      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32, is_training=True)
validation_generator = generator(validation_samples, batch_size=32, is_training=False)

import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Activation

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
'''
model.add(Convolution2D(3, 5, 5))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
'''
model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(48, 3, 3))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(10))
#model.add(Activation('relu'))

model.add(Dense(1))

print(model.summary())

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
	validation_data=validation_generator, nb_val_samples=len(validation_samples), 
	nb_epoch=30, verbose=1)

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

model.save('model.h5')
