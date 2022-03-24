'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
__all__ =['resnet18A_1w1a','resnet18B_1w1a','resnet18C_1w1a','resnet18_1w1a']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.nonlinear = nn.Tanh()
     
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
      
        self.nonlinear2 = nn.Tanh()
     

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = (self.nonlinear((self.bn1(self.conv1(x))))) 
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = (self.nonlinear2((out)))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        chs = 1
        num_channel2 = [(2*0.9141),(2*0.8594),(2*0.8754),(2*0.9082),(2*0.8203)]
        #num_channel2 = [int(2*0.875),int(2*0.8984),int(2*0.8828),int(2*0.918),int(2*0.605)]
        self.in_planes = int(num_channel[0]*num_channel2[0])

        self.conv1 = BinarizeConv2d(3, int(num_channel[0]*num_channel2[0]), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(num_channel[0]*num_channel2[0]))
        
        self.nonlinear = nn.Tanh()
        
        self.layer1 = self._make_layer(block, int(num_channel[0]*num_channel2[1]), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(num_channel[1]*num_channel2[2]), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(num_channel[2]*num_channel2[3]), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(num_channel[3]*num_channel2[4]), num_blocks[3], stride=2)
        self.linear = bilinear(int(num_channel[3]*num_channel2[4])*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(int(num_channel[3]*num_channel2[4]*block.expansion))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlinear((self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out 


def resnet18A_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18B_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[32,64,128,256],**kwargs)

def resnet18C_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[64,128,256,512],**kwargs)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])