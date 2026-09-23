# Note :
# Traing is done on kaggle because i have no GPU.
# and the dataset is from kaggle datasets.
 # The training data is in the form of images of cats and dogs.
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense,Flatten , Conv2D, MaxPooling2D,Dropout,BatchNormalization,Activation
from keras.utils import image_dataset_from_directory as loader 
# loading the data 
train_ds =  loader(
    directory = '/kaggle/input/datasets/melvinpauljacob/cat-dog-dataset/dogscats/images',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256)
)

test_ds =  loader(
    directory = '/kaggle/input/datasets/melvinpauljacob/cat-dog-dataset/dogscats/valid',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256)
)
#normalizinf the data 

def process(image,lable):
    image = tf.cast(image/256,tf.float32)
    return image , lable
train_ds = train_ds.map(process)
test_ds = test_ds.map(process) 

# creating our CNN 
model = Sequential()
#frist convolution layer with max pooling 
model.add(Conv2D(32, padding = 'valid',kernel_size = (3,3),input_shape = (256,256,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
# secound layer
model.add(Conv2D(32, padding='same',activation='relu',kernel_size = (3,3)),activation='relu')
model.add(Dropout(0.25))
# third layer 
model.add(Conv2D(64,activation='relu'))
# fourth layer 
model.add(Conv2D(128, padding='valid',activation='relu',kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))   
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Flatten())
# adding layers 
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history =  model.fit(train_ds,validation_data=test_ds,epochs=10)

import matplotlib.pyplot as plt
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
plt.show()