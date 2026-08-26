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
import keras
from keras import Sequential
from keras.layers import Dense 

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
print(x_train)