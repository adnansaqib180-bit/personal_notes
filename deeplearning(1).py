from sklearn.linear_model import Perceptron

data = {
    'num_bedrooms' : [2, 3, 4, 5, 6],
    'sqft' : [1000, 1500, 2000, 2500, 3000],
    'price' : [200000, 250000, 300000, 350000, 400000]
}

import pandas as pd 

df = pd.DataFrame(data)

x = df.drop(columns=['price'])

y = df['price']

model = Perceptron()
model.fit(x,y)

df = pd.DataFrame({
    'num_bedrooms' : [10],
    'sqft' : ['3200']
})

print(model.predict(df))

