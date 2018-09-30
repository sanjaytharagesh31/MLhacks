import numpy as np 
import pandas as pd
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.layers import Flatten 
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils 

seed = 7
np.random.seed(seed)
image_size = None

# Load training data from a file and return X, y
def load_training_data(file_name):
    raw_data = genfromtxt(file_name, delimiter=',', skip_header=1)

    raw_sample_image = raw_data[0][1:]
    #print(raw_sample_image)
    image_size = int(np.sqrt(len(raw_sample_image)))
    #print(image_size)
    X_shape = (len(raw_data), image_size, image_size, 1)
    y_shape = (len(raw_data), 10)

    X_data = np.zeros(X_shape)
    y_data = np.zeros(y_shape)
    for index, datum in enumerate(raw_data):
        X_data[index] = np.array(datum[1:]/255).reshape(image_size, image_size, 1)
        y_data[index] = np_utils.to_categorical(int(datum[0]), 10)       #one-hot encoding 

    return X_data, y_data

# Load evaluation data from a file and return X
def load_eval_data(file_name):
    raw_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)

    raw_sample_image = raw_data[0]
    image_size = int(np.sqrt(len(raw_sample_image)))
    X_shape = (len(raw_data), image_size, image_size, 1)

    X_data = np.zeros(X_shape)
    for index, datum in enumerate(raw_data):
        X_data[index] = np.array(datum/255).reshape(image_size, image_size, 1)

    return X_data 


print("Loading training data")
X_train, y_train = load_training_data("train.csv")

#create model
model = Sequential()
model.add(Conv2D(30, (5,5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#fit the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=1, shuffle=True)

X_eval = load_eval_data("test.csv");
output_file = "output.csv"
with open(output_file, 'w') as f:
    f.write('ImageId,Label\n')
    y_eval = model.predict(X_eval)
    for index, y_hat in enumerate(y_eval):
        prediction = np.argmax(y_hat)
        f.write(str(index+1)+','+str(prediction)+'\n')
    f.close()
