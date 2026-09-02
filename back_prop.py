import numpy as np 
import pandas as pd 

data = {
    'sqft' : [1000],
    'price' : [200000]
}
df= pd.DataFrame(data)
x = df['sqft']
y = df['price']
def model(g,act):
    w = 1
    b = 1
    for _ in range(20):

        for x in g:
            y_ht = w*x + b
            for y in act:
                loss = (y-y_ht)**2
                w = w-(0.00001)*(-2*(y-y_ht)*(x))
                b = b-(0.00001)*(-2*(y-y_ht)*(x))
            print('l',loss)
            print('y',y,'p',y_ht)
            print(w,b)
model(x,y)