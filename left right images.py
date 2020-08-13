# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:23:00 2020

@author: habibur nobel
"""

#Part1 - Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

#Step1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step2 - MaxPooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step3 - Flattening
classifier.add(Flatten())

#Step4 - Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Part2 - Fitting CNN into images:
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('leftrightimages/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('leftrightimages/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit(
        training_set,
        steps_per_epoch=9642,
        epochs=5,
        validation_data=test_set,
        validation_steps=2410)






