import numpy as np 
import pandas as pd 

data = {
    'sqft' : [1000, 1500, 2000, 2500, 3000],
    'price' : [200000, 250000, 300000, 350000, 400000]
}
df = pd.DataFrame(data)
x = df['sqft']
y = df['price']

def model(x, y):
    w = 0.1  
    b = 0.1
    learning_rate = 0.0000001
    for single_x, single_y in zip(x, y):
        
        y_ht = w * single_x + b
        
        loss = (single_y - y_ht) ** 2
        
        dj_dw = -2 * single_x * (single_y - y_ht)
        dj_db = -2 * (single_y - y_ht)
        
        w = w - (learning_rate * dj_dw)
        b = b - (learning_rate * dj_db)
        
        print(f"Sqft: {single_x} | Actual Price: {single_y} | Predicted: {y_ht:.2f} | Loss: {loss:.2f}")

model(x, y)
