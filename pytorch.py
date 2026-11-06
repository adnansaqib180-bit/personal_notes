from torch.nn import *

class Model(Module):
    def __init__(self, features):
        super().__init__()
        self.network = Sequential(
            Linear(features,3),
            ReLU(),
            Linear(3,1),
            Sigmoid()
        )
    def forward(self,features):
        out = self.network(features)
        return out 
model =  Model(12)
loss = BCELoss()
from torch import optim
optimizer = optim.Adam(model.parameters(),0.001)
epochs = 100
for i in range(epochs):
    loss = 0
    optimizer.zero_grad()
    pred = model(12)
    los = loss(pred,1)
    los.backwoard()
    los.step()

    loss+= los.item()
    print(loss/12)


# cnn using torch 

class CNN(Module):
    def __init__(self, num_channels):
        super().__init__()
        self.cnn = Sequential(
            Conv2d(num_channels,32,2,padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(num_channels,32,2,padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2)

        )
        self.fnn = Sequential(
            Flatten(),
            Linear(32*14*4,64),
            ReLU(),
            Linear(128,10)
        )