import numpy as np 
import pandas as pd 

data = {
    'sqft' : [1000, 1500, 2000],
    'price' : [200000, 250000, 300000]
}
df= pd.DataFrame(data)
x = df['sqft']
y = df['price']
def model(x,y):
    for x in x:
        w = 1
        b = 1
        y_ht = w*x + b
        loss = (y-y_ht)**2
        w = w*(0.1)*(-2*(y-y_ht)*(y_ht))
        b = b*(0.1)*(-2*(y-y_ht)*(y_ht))
        print('l',loss)
        print('y',y,'p',y_ht)
model(x,y)