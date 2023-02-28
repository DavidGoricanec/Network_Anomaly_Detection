import torch.nn as nn
import config

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(config.col_length, config.col_length)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(config.col_length, config.col_length)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(config.col_length, config.col_length)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(config.col_length, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
