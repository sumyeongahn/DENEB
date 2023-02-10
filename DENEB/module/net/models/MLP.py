import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.Block = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, num_classes)


    def extract(self, x):
        x = x.view(x.size(0), -1)
        feat = self.Block(x)
        return feat


    def predict(self, x):
        prediction = self.fc(x)
        return prediction
        
    def forward(self, x):
        x = self.extract(x)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit