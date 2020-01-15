#Import cifar10 dataset
from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils

from matplotlib import pyplot as plt
from PIL import Image

import h5py

#10 Categories of image
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Array of training and testing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

new_x_train = x_train.astype('float32')
new_x_test = x_test.astype('float32')

new_x_train /= 255
new_x_test /= 255

new_y_train = np_utils.to_categorical(y_train)
new_y_test = np_utils.to_categorical(y_test)

#Image recognition model used
#Linear stack of layer
model = Sequential()
#Conventual 2D Image
#Image is 32x32, 3x3 block
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='tanh', padding='same', kernel_constraint=maxnorm(3)))
#Make sure all pixel read for more accurate guess
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
#Distinguish image detail
model.add(Dense(512,activation='tanh', kernel_constraint=maxnorm(3)))
#Prevent over-detail; kept new info to be added
model.add(Dropout(0.5))
#Improve accuracy & use alternative activation
model.add(Dense(10, activation='softmax'))

#Set loss optmizer, set what data desired (accuracy)
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

history = model.fit(new_x_train, new_y_train, epochs=50, batch_size=32)

model.save('trained_model.h5')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()