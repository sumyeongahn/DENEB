import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self,  num_classes):
        super(PreActResNet, self).__init__()
        num_blocks = [2,2,2,2]
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(PreActBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*PreActBlock.expansion, num_classes)

    def _make_layer(self, PreActBlock, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def extract(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
        return out

    def predict(self,x):
        out = self.fc(x)
        return out
    
    def forward(self,x):
        x = self.extract(x)
        x = x.view(x.size(0),-1)
        x = self.predict(x)
        return x



def PreActResNet18(num_classes, double_fc):
    return PreActResNet(num_classes, double_fc)
