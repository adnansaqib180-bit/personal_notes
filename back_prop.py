import numpy as np 
# import pandas as pd 

# data = {
#     'sqft' : [1000],
#     'price' : [200000]
# }
# df= pd.DataFrame(data)
# x = df['sqft']
# y = df['price']
# def model(g,act):
#     w = 1
#     b = 1
#     for _ in range(20):

#         for x in g:
#             y_ht = w*x + b
#             for y in act:
#                 loss = (y-y_ht)**2
#                 w = w-(0.00001)*(-2*(y-y_ht)*(x))
#                 b = b-(0.00001)*(-2*(y-y_ht)*(x))
#             print('l',loss)
#             print('y',y,'p',y_ht)
#             print(w,b)
# model(x,y)

# import numpy as np


# def softmax(num):
#     n = np.exp(num - np.max(num, axis=-1, keepdims=True))
#     result = n / np.sum(n, axis=-1, keepdims=True)
#     return result 



# def self_attention(*X):
#     X = np.array(X)
#     XT = np.transpose(X)
#     XXT = np.dot(X,XT)
#     result = np.dot(softmax(XXT),X)
#     return result

# a = [1,2,3,4]
# b = [5,6,7,8]

# print(self_attention(a,b))
# # = softmax(KQ)V 
# def advanced_SA(*X):
#     X = np.array(X)
#     weights = np.array([[1,2,3],[2,3,4],[3,4,5]])
#     K,Q,V = np.dot(X,weights)
#     KQ = np.dot(K,Q)
#     result = np.dot(softmax(KQ),V)
#     return result 
# c = [1,2,3]
# d = [5,6,7]
# print(advanced_SA(c,d))

text = '''hello my dear '''
text = text.strip()
text = text.split(sep=' ')
print(text)
from keras.layers import *
from keras import Sequential
X = np.array([1,2,3])
model = Sequential()
model.add(Embedding(3,10))
model.add(Dense(1,'linear'))
model.compile('adam','mse')
model.fit(X)
x = model.predict(X)
print(x)