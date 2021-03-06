import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Pooll(nn.Module):

    def __init__(self):
        super(Pooll, self).__init__()
        self.pooll = nn.AvgPool2d(4)

    def forward(self, x):
        '''
        TODO: the first dimension is the data batch_size
        so we need to decide how the input shape should be like
        '''
        return self.pooll(x[0])

class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()
        self.a = 1

    def forward(self, input):
        '''
        TODO: the first dimension is the data batch_size
        so we need to decide how the input shape should be like
        '''
        return input.view(input.size(0), -1)

class CustomNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomNetwork, self).__init__()
        self.linear = nn.Linear(512*4, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


class ConvBlk(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, pad=1, stride=1):
        super(ConvBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in,
                              out_channels=c_out,
                              kernel_size=k_size,
                              padding=pad,
                              stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        in_f = 16
        self.conv1 = ConvBlk(1, in_f, k_size=3, pad=1, stride=2)
        self.convsp = ConvBlk(in_f, in_f, k_size=1, pad=1, stride=1)
        self.conv2 = ConvBlk(in_f, in_f*2, k_size=3, pad=1, stride=2)
        self.conv3 = ConvBlk(in_f*2, in_f*4, k_size=3, pad=0, stride=2)
        self.conv4 = ConvBlk(in_f*4, in_f*8, k_size=3, pad=0, stride=1)
        self.conv5 = ConvBlk(in_f*8, 10, 1, 0)
        self.do1 = nn.Dropout2d(0.4)
        self.do2 = nn.Dropout2d(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.convsp(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.do1(x)
        x = self.conv4(x)
        x = self.do2(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, iden='na'):
        super(Bottleneck, self).__init__()

        self.id = iden

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu3 = nn.ReLU()

        #self.shortcut = nn.Sequential()
        self.shortcutconv1 = nn.Conv2d(in_planes, self.expansion*planes,
                      kernel_size=1, stride=stride, bias=False)
        self.shortcutbn1 = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        x = x[0]
        results = OrderedDict()
        ii = 0
        out = self.relu1(self.bn1(self.conv1(x)))
        results[str(self.id)+str(ii)] = out
        ii += 1
        out = self.relu2(self.bn2(self.conv2(out)))
        results[str(self.id)+str(ii)] = out
        ii += 1
        out = self.bn3(self.conv3(out))
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            convx = self.shortcutconv1(x)
            convx = self.shortcutbn1(convx)
            out += convx
        else:
            #self.shortcut(x)
            out += x
        out = self.relu3(out)
        results[str(self.id)+str(ii)] = out
        ii += 1
        return out, results


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, iden='1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, iden='2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, iden='3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, iden='4')
        self.pooll = Pooll()
        self.viewl = View()
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, iden):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList([])
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, iden+str(i)))
            i += 1
            self.in_planes = planes * block.expansion
        #return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        ii = 0
        results = OrderedDict()
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        results['d'+str(ii)] = out
        ii += 1
        for layer in self.layer1:
            out, rr = layer((out, None))
            results.update(rr)
        #out = self.layer1(out)
        for layer in self.layer2:
            out, rr = layer((out, rr))
            results.update(rr)
        for layer in self.layer3:
            out, rr = layer((out, rr))
            results.update(rr)
        for layer in self.layer4:
            out, rr = layer((out, rr))
            results.update(rr)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = self.pooll((out, rr))
        out = self.viewl(out)
        results['d'+str(ii)] = out
        ii += 1
        out = self.linear(out)
        results['d'+str(ii)] = out
        ii += 1
        return out, results

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.viewl = View()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        results = OrderedDict()
        out = x
        ii = 0
        for layer in self.features:
            out = layer(out)
            results['c'+str(ii)] = out
            ii += 1
        #out = self.features(x)
        #out = out.view(out.size(0), -1)
        out = self.viewl(out)
        out = self.classifier(out)
        results['l'+str(ii)] = out
        ii += 1
        return out, results

    def _make_layers(self, cfg):
        layers = nn.ModuleList([])
        in_channels = 3
        ii = 0
        for x in cfg:
            if x == 'M':
                layers.extend([nn.MaxPool2d(kernel_size=2, stride=2)])
            else:
                layers.extend([nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)])
                in_channels = x
        layers.extend([nn.AvgPool2d(kernel_size=1, stride=1)])
        #return nn.Sequential(*layers)
        return layers
