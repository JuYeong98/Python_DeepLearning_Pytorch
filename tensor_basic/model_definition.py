from torch.nn import Linear
from torch.nn import Module
import torch
#단순 신경망을 정의하는 방법
model = Linear(in_features = 1, out_features=1 , bias = True)

#nn.Module()을 상속하여 정의하는 방법
class MLP(Module):
    def __init__(self , inputs):
        super(MLP , self).__init__()
        self.layer = Linear(inputs , 1)
        self.activation  = torch.Sigmoid()
    
    def forward(self , X):
        X  = self.layer(X)
        X  = self.activation(X)
        
        return X


##Sequential 신경말을 정의하는 방법

import torch.nn as nn
class Sequential_custom(nn.Module):
    def __init__(self):
        super(Sequential_custom, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=5),
            nn.ReLU(inplace =  True),
            nn.MaxPool2d(2))    
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64 , out_channels=30, kernel_size=5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5 ,out_features=10, bias = True),
            nn.ReLU(inplace = True)
        )
        
        def forward(self,x):
            x= self.layer1(x)
            x= self.layer2(x)
            x= x.view(x.shape[0], -1)
            x= self.layer3(x)
            return x
        
model  = Sequential_custom()

print('Printing children\n---------------------------')
print(list(model.children))
print('Printing Modules\n---------------------------')
print(list(model.modules))
