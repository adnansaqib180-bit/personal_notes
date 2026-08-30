import keras
from keras import Sequential
from keras.layers import Dense , Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,precision_score
import warnings
warnings.filterwarnings("ignore")

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

print(x_train.shape)
print(x_test.shape)

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(392,activation='relu'))
model.add(Dense(196,activation='relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

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






