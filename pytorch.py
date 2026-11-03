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
