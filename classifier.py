from __future__ import print_function

import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys

from IPython.display import display, Image
from scipy import ndimage
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras import optimizers
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn import linear_model

##This notebook is built around using tensorflow as the backend for keras 
#!pip install pillow
#!KERAS_BACKEND=tensorflow python -c "from keras import backend"

##Dimensions of our images.# dimen 
img_width, img_height = 150, 150

# train_data_dir = 'data/train'
# validation_data_dir = 'data/validation'

# ##Used to rescale the pixel values from [0, 255] to [0, 1] interval
# datagen = ImageDataGenerator(rescale=1./255)

# ##Automagically retrieve images and their classes for train and validation sets
# train_generator = datagen.flow_from_directory(
#         train_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=16,
#         class_mode='binary',
#         classes = ['neutral', 'sadness', 'surprise', 'happiness', 'fear', 'anger', 'contempt', 'disgust'])

    # validation_generator = datagen.flow_from_directory(
    #         validation_data_dir,
    #         target_size=(img_width, img_height),
    #         batch_size=32,
    #         class_mode='binary',
    #         classes = ['neutral', 'sadness', 'surprise', 'happiness', 'fear', 'anger', 'contempt', 'disgust'])

model = 0

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
def train():

    print("CREATE FUNCTION THAT TRAINS A MODEL AND SAFE THE MODEL IN A GLOBAL VARIABLE")

        ## For future you might want to be able to save and load models (Then you don't have to retrain your model 
        ## every time your program starts).

    with open("fer2013.csv") as f:
        content = f.readlines()
    
        lines = np.array(content)
    
        num_of_instances = lines.size
        print("number of instances: ",num_of_instances)

    filname='fer2013.csv'
    def getData(filname):
        # images are 48x48 = 2304 size vectors
        # N = 35887
        Y = []
        X = []
        first = True
        for line in open(filname):
            if first:
                first = False
            else:
                row = line.split(',')
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])

        X, Y = np.array(X) / 255.0, np.array(Y) # scaling is already done here 
        return X, Y 

    X, Y = getData(filname)
    print(X.shape)
    print(Y.shape)
    print(len(set(Y)))
    num_class=len(set(Y))

    #check if classes are balanced
    def balance_class(Y): 
        num_class=set(Y)
        count_class={}
        for i in range(len(num_class)):
            count_class[i]=sum([1 for y in Y if y==i])
        return count_class

    balance=balance_class(Y)

    #reshape X to fit keras with tensorflow backend
    N,D = X.shape
    X = X.reshape(N,48,48,1) # last dimension =1 is because it is black and white image, if colored, it will be 3

    #split into training and testing set and rearrange the label y into 7 classes 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    y_train= (np.arange(num_class) == y_train[:,None]).astype(np.float32)
    y_test=(np.arange(num_class) == y_test[:,None]).astype(np.float32)

    X_train, X_test, y_train, y_test

    num_classes = 7

    model = Sequential()
    
    #1st convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    
    #2nd convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    
    #3rd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    
    model.add(Flatten())
    
    #fully connected neural networks
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(num_classes, activation='sigmoid'))

    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                optimizer= rmsprop,
                metrics=['accuracy'])

    ##Train Model
    nb_epoch = 1
    nb_train_samples = 1
    nb_validation_samples = 400

    model.fit(X_train, y_train, 
                steps_per_epoch=nb_train_samples, 
                epochs=nb_epoch, 
                verbose=1, 
                validation_data=(X_test, y_test),
                validation_steps=nb_validation_samples)

    model.evaluate(X_test, y_test, verbose=0) 

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1]) 

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'




def predict(path):

    print("CREATE FUNCTION THAT PREDICTS FOR THE IMAGE IN PATH path AND RETURN RESULT")
    # return 0

    from keras.preprocessing import image
    name_img=path#'angry.jpeg'
    test_image=image.load_img(name_img, target_size =(48,48))
    test_image=image.img_to_array(test_image)
    test_image.shape

    def rgb2gray(rgb): # turn the image into gray instead of having 3 colors

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
        
    test_image=rgb2gray(test_image)
    test_image.shape

    def new_img_convert(img):
        img=img.reshape((48,48,1))
        img/=255
        img=np.expand_dims(img, axis=0)
        return img

    img=new_img_convert(test_image)
    img.shape

    model =load_model('my_model.h5')
    predictions=model.predict(img)
    result = label_map[get_max_index(predictions[0])]
    print(predictions)
    print(result)
    return result
    # print (label_map)
    # print(result)


def get_max_index(list):
    max_value = list[0]
    max_i = 0
    for i in range(1, len(list)):
        if list[i] > max_value:
            max_value = list[i]
            max_i = i

    return max_i