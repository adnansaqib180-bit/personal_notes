# LeNet CNN model

from keras.models import Sequential
from keras.layers import  Dense , Flatten , Conv2D , AveragePooling2D 

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2,padding='valid'))
model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh',padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2,padding='valid'))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))

print(model.summary())