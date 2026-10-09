# LeNet CNN model

from keras.models import Sequential
from keras.layers import *

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

# simple RNN layer

model.add(SimpleRNN(150, activation='tanh', return_sequences=True))
# then simply add a TimeDistributed layer to apply the Dense layer to each time step
model.add(TimeDistributed(Dense(10, activation='softmax')))

# LSTM layer

model.add(LSTM(150, activation='tanh', return_sequences=True))
# then simply add a TimeDistributed layer to apply the Dense layer to each time step
model.add(TimeDistributed(Dense(10, activation='softmax')))

# Gru layer 
model.add(GRU(150,activation='tanh',return_sequences=True))
# then simply add a TimeDistributed layer to apply the Dense layer to each time step
model.add(TimeDistributed(Dense(10, activation='softmax')))    

# deep rnn

model.add(SimpleRNN(150, activation='tanh', return_sequences=True))
model.add(SimpleRNN(150, activation='tanh', return_sequences=True))

# this is a deep rnn with 2 layers 
# you can also use LSTM or GRU instead of SimpleRNN

#bidirectional RNN

model.add(Bidirectional(SimpleRNN(150, activation='tanh', return_sequences=True)))

# again you can use LSTM or GRU instead of SimpleRNN 