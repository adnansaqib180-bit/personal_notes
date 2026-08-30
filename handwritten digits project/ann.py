import keras
# from keras import Sequential
# from keras.layers import Dense , Flatten

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255
y_train = y_train/255

print(x_train.shape)
print(y_train.shape)

# model = Sequential()
# model.add(Flatten())
