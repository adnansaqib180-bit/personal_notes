import pandas as pd 
from keras import Sequential
from keras.layers import Dense
import matplotlib.pylab as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import keras_tuner as kt 
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('personal_notes/DL (ANN) projects/new_hr.csv')
print(df.head())
print(df.columns)
print(df.isnull().sum())
print(df.info())

df = df.drop(columns=['Attrition','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','MaritalStatus','RelationshipSatisfaction'])

df['OverTime'] = df['OverTime'].map({"Yes":1,'No':0})
df['Over18'] = df['Over18'].map({"Y":1,'N':0})
df['Gender'] = df['Gender'].map({'Male':1,"Female":0})
df['BusinessTravel']  =  df['BusinessTravel'].map({"Non-Travel":0,'Travel_Rarely':1,'Travel_Frequently':2})
df = pd.get_dummies(data=df,columns=['Department','EducationField','JobRole'],drop_first=True,dtype=int)

print(len(df.columns))
print(df.info())

x = df.drop(columns=['MonthlyIncome'])
y = df['MonthlyIncome']

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)

def build_model(hp):
    model = Sequential()
    nodes = hp.Int('nodes', min_value=32, max_value=256, step=32)
    activ = hp.Choice('activation',values=['relu','sigmoid'])
    model.add(Dense(units=nodes,activation=activ))
    model.add(Dense(1, activation='linear')) 
    optimizer = hp.Choice('optimizer',values=['adam','sgd'])
    model.compile(optimizer= optimizer,loss='mse',metrics=['mae'])
    return model
tuner = kt.RandomSearch(build_model,objective='val_mae',max_trials=5)
tuner.search(X_train,y_train,epochs=10,validation_data=(X_test,y_test))
print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models=1)[0]
print(model.summary())

history = model.fit(X_train,y_train,epochs=500,validation_split = .24)

predictions = model.predict(X_test)
print("r2 Score:", r2_score(y_test, predictions))


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train Accuracy")
plt.plot(history.history["val_mae"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()

# prety good 93% r2 score
