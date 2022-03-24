'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from modules import *


__all__ = ['vgg_small_1w1a']


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.classifier(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Hardtanh(inplace=True)]
            else:
                layers += [conv2d, nn.Hardtanh(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


#chs = [3*0.8958,3*0.8958,3*0.8893,3*0.8945,3*0.7891,3*0.6237]
class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10,channels=3):
        super(VGG_SMALL_1W1A, self).__init__()
        chs = [3*0.8984,3*0.9167,3*0.8945,3*0.8854,3*0.8112,3*0.6341]
        #chs = [3,3,3,3,3,3]
        self.conv0 = BinarizeConv2d(3, int(128*chs[0]), kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(128*chs[0]))
        self.conv1 = BinarizeConv2d(int(128*chs[0]), int(128*chs[1]), kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(int(128*chs[1]))
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = BinarizeConv2d(int(128*chs[1]), int(256*chs[2]), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(256*chs[2]))
        self.conv3 = BinarizeConv2d(int(256*chs[2]),int(256*chs[3]), kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(256*chs[3]))
        self.conv4 = BinarizeConv2d(int(256*chs[3]), int(512*chs[4]), kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(512*chs[4]))
        self.conv5 = BinarizeConv2d(int(512*chs[4]),int( 512*chs[5]), kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(512*chs[5]))
        self.fc = bilinear(int(15568), num_classes) # for vgg this number needs to be manually adjusted #8192*chs[5] 15568
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
       
      
        x = self.fc(x)
        return x



def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model
