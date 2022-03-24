'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *

from torch.autograd import Variable

__all__ = ['resnet20_1w1a']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class adapter2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(adapter, self).__init__()
        # == channel attention
        self.stride = 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att_channel = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

        # == spatial attention

        self.down_sample = nn.AvgPool2d(2, stride=2)
        self.att_spatial =  nn.Conv2d(channel, channel,  kernel_size=3, padding=1, groups=channel, bias=True)


    def forward(self, x):

        # == first part channel attention
        
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # print(x.mean())
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # 1D conv
        y = self.att_channel(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.att
        #Multi-scale information fusion
        y = self.sigmoid(y)
        y1 = down

        
    
    
        return  y+x

class adapter(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(adapter, self).__init__()
        # == channel attention
        self.stride = 1
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.att_channel = BinarizeConv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        #self.sigmoid = nn.Sigmoid()
        
        # == spatial attention
        self.att_spatial =  BinarizeConv2d(channel, 3*channel,  kernel_size=1,padding=0,  bias=False,mask = False)
        self.att_spatial1 =  BinarizeConv2d(3*channel, channel,  kernel_size=1,padding = 0 ,bias=False,mask = False)
        #self.att_spatial2 =  BinarizeConv2d(3*channel, 3*channel,  kernel_size=3, padding=1, groups=channel, bias=False,mask = False)
        #self.att_spatial3 =  BinarizeConv2d(3*channel, channel,  kernel_size=3, padding=1, groups=channel, bias=False,mask = False)
        
        
        self.btn = nn.BatchNorm2d(3*channel)
        self.btn1 = nn.BatchNorm2d(channel)
        #self.btn2 = nn.BatchNorm2d(3*channel)
        #self.btn3 = nn.BatchNorm2d(channel)

    def forward(self, x):

        # == first part channel attention
        
        
        #b, c, h, w = x.size()
  
       
        #y = self.avg_pool(x)
        # 1D conv
        #y = self.att_channel(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
      
        #y = self.sigmoid(y)
        y1 = self.att_spatial(x)
        y1 = (self.btn(y1))
        
        y2 = self.btn1(self.att_spatial1(y1))
       

        #y3 = self.att_spatial2(y2)
        #y3 = F.hardtanh(self.btn2(y3))

        #y4 = self.att_spatial3(y3)
        #y4 = F.hardtanh(self.btn3(y4))
        
    
        return  y2 + x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
    

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     BinarizeConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.hardtanh(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,channels= 1):
        super(ResNet, self).__init__()
        self.chs = [3,3,3*0.9792,3*0.9688]
        self.in_planes = int(16*self.chs[0])
        self.conv1 = BinarizeConv2d(3, int(16*self.chs[0]), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16*self.chs[0]))
        self.layer1 = self._make_layer(block, int(16*self.chs[1]), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32*self.chs[2]), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64*self.chs[3]), num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(int(64*self.chs[3]))
        self.linear = bilinear(int(64*self.chs[3]), num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1e-8)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks-1) 
        layers = [] 
        for stride in strides:
            
            layers.append(block(self.in_planes, planes, stride)) 
            self.in_planes = planes * block.expansion 
            num=1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1) 
        out = self.bn2(out)
        out = self.linear(out)

        return out


def resnet20_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [3, 3, 3],**kwargs)


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()