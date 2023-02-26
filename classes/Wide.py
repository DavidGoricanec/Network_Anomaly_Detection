class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(116, 116*3)
        self.relu = nn.ReLU()
        self.output = nn.Linear(116*3, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
 