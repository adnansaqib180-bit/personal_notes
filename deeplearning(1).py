# from sklearn.linear_model import Perceptron

# data = {
#     'num_bedrooms' : [2, 3, 4, 5, 6],
#     'sqft' : [1000, 1500, 2000, 2500, 3000],
#     'price' : [200000, 250000, 300000, 350000, 400000]
# }

# import pandas as pd 

# df = pd.DataFrame(data)

# x = df.drop(columns=['price'])

# y = df['price']

# print('started ..')
# model = Perceptron(max_iter=10000000)
# model.fit(x,y)

# df = pd.DataFrame({
#     'num_bedrooms' : [10],
#     'sqft' : ['3200']
# })

# print(model.predict(df))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,precision_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist 

import numpy as np
import pandas as pd

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_test = x_test/255
x_train = x_train/255

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Flatten())
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(392,activation='relu'))
model.add(Dense(196,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=2,validation_split=.2)

y_prob = model.predict(x_test)
predictions = y_prob.argmax(axis=1)

print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions,average='weighted'))
print("Accuracy Score:", accuracy_score(y_test, predictions))
print('Precision : ',precision_score(y_test, predictions,average='weighted'))


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()