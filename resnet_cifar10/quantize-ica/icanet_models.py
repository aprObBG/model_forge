import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import *
import random
import numpy.random as rd
import numpy.linalg as la


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

class ConvBNCELU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNCELU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.CELU(inplace=True),
        )

class ConvBlk(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, pad=1, stride=1, config=''):
        super(ConvBlk, self).__init__()
        self.conv = nn.Conv2dQuant(in_channels=c_in,
                              out_channels=c_out,
                              kernel_size=k_size,
                              padding=pad,
                              stride=stride,
                              config=config)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class PConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=None, config=''):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(PConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=None, config=''):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class QuantConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=None, config=''):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(QuantConvBNReLU, self).__init__(
            Conv2dQuant(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False, config=config),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            QuantLayer(config=config)
        )


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

    def __init__(self, in_planes, planes, stride=1, iden='na', dl=16, config=''):
        super(Bottleneck, self).__init__()

        self.id = iden

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = Conv2dQuant(in_planes, planes, kernel_size=1, bias=False, config=config)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.quant1 = QuantLayer(config=config)
        self.conv2 = Conv2dQuant(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, config=config)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.quant2 = QuantLayer(config=config)
        self.conv3 = Conv2dQuant(planes, self.expansion *
                               planes, kernel_size=1, bias=False, config=config)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu3 = nn.ReLU()
        self.quant3 = QuantLayer(config=config)

        #self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            self.shortcutconv1 = Conv2dQuant(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, config=config)
            self.shortcutbn1 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcutquant1 = QuantLayer(config=config)

    def forward(self, x):
        x = x[0]
        results = OrderedDict()
        ii = 0
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.quant1(out)
        results[str(self.id)+str(ii)] = out
        ii += 1
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.quant2(out)
        results[str(self.id)+str(ii)] = out
        ii += 1
        out = self.bn3(self.conv3(out))
        out = self.quant3(out)
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            convx = self.shortcutconv1(x)
            convx = self.shortcutbn1(convx)
            out += convx
        else:
            #self.shortcut(x)
            out += x
        out = self.relu3(out)
        out = self.shortcutquant1(out)
        results[str(self.id)+str(ii)] = out
        ii += 1
        return out, results


class Neg_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Neg_Entropy_Loss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b

class Decoder(nn.Module):
    def __init__(self, dl, num_classes=10, config=''):
        super(Decoder, self).__init__()
        self.config = config
        self.cls = nn.Sequential(
            QuantConvBNReLU(384, 256, kernel_size=3, padding=1, stride=1, config=config),
            QuantConvBNReLU(256, 256, kernel_size=3, padding=1, stride=2, config=config),
            QuantConvBNReLU(256, 128, kernel_size=3, padding=1, stride=1, config=config),
        )
        #self.transconv = ConvTranspose2dQuant(A[ii], S[ii], stride=self.trans_stride, padding=self.trans_padding)
        #ConvTranspose3dQuant(out_feat, int(out_feat/2), kernel_size=(2, 2, 2), stride=(2, 2, 2), config=config
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.lr = LinearQuant(512, num_classes, config=config)
        self.trans_stride = 2
        self.trans_padding = 0
        self.dl = dl

    def forward(self, A, S):
        if self.config["quantization"].lower() == "fixed":
            A = quant.fixed_nn(A, self.config["activation_i_width"], self.config["activation_f_width"])
            S = quant.fixed_nn(S, self.config["activation_i_width"], self.config["activation_f_width"])
        else:
            A = A
            S = S
        print('decoder A', A.shape)
        print('decoder S', S.shape)
        A = A.reshape(A.shape[0], -1, self.dl, *A.shape[-2:])
        x = torch.empty(A.shape[0], A.shape[1], 3, S.shape[-2]+2, S.shape[-1]+2, dtype=A.dtype).to(A.device)
        for ii in range(A.shape[0]):
            x[ii] = F.conv_transpose2d(A[ii], S[ii], stride=self.trans_stride, padding=self.trans_padding)
        if self.config["quantization"].lower() == "fixed":
             x = quant.fixed_nn(x, self.config["activation_i_width"], self.config["activation_f_width"])
        else:
             x = x

        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        print('transed x', x.shape)
        x = self.cls(x)
        print('after cls x', x.shape)
        x = self.pool(x)
        print('after pool x', x.shape)
        x = x.view(x.shape[0], -1)
        x = self.lr(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dl, config=''):
        super(Encoder, self).__init__()
        self.dl = dl

        self.route_AS = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, padding=1, stride=1),
            ConvBNReLU(32, 64, kernel_size=3, padding=1, stride=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1, stride=2),
        )
        self.route_A = nn.Sequential(
            ConvBNReLU(32, self.dl, kernel_size=1, padding=0, stride=1),
        )
        self.route_S = nn.Sequential(
            ConvBNReLU(32, self.dl*3, kernel_size=3, padding=1, stride=1),
            ConvBNReLU(self.dl*3, self.dl*3, kernel_size=3, padding=1, stride=2),
        )
        self.trans_padding = 3
        self.trans_stride = 2
        self.recon_loss = nn.MSELoss()
        self.neg_entropy = Neg_Entropy_Loss()

    def forward(self, x):
        AS = self.route_AS(x)
        A, S = torch.chunk(AS, 2, 1)
        A = self.route_A(A)
        S = self.route_S(S)
        b, _, pa, _ = A.shape
        _ = None
        _, _, ps, _ = S.shape
        S = S.reshape(b, self.dl, 3, ps, ps)
        if self.training:
            # S = S.unsqueeze(2)
            A = A.unsqueeze(1)
            recon = torch.empty(b, 3, x.shape[-2], x.shape[-1], dtype=x.dtype).to(x.device)
            for ii in range(b):
                recon[ii] = F.conv_transpose2d(A[ii], S[ii], padding=self.trans_padding, stride=self.trans_stride)
            img = torch.sigmoid(x)
            recon = torch.sigmoid(recon)
            l1 = self.recon_loss(img, recon)
            l2 = torch.relu(S).mean()
            l3 = self.neg_entropy(S)
            A = A.squeeze(1)
            S = S.squeeze(2)
            return A, S, 0.1*l1, 0.1*l2, 0.1*l3
        return A, S, 0, 0, 0


class ICA_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, config=''):
        super(ICA_ResNet, self).__init__()
        # print('config', config)
        self.dl = 16
        self.in_planes = 64
        self.conv1 = Conv2dQuant(self.dl, 64, kernel_size=3, stride=1, padding=1, bias=False, config=config)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, iden='1', dl=self.dl, config=config)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, iden='2', dl=self.dl, config=config)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, iden='3', dl=self.dl, config=config)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, iden='4', dl=self.dl, config=config)
        self.pooll = Pooll()
        self.viewl = View()
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.encoder = Encoder(dl=self.dl, config=config)
        self.decoder = Decoder(dl=self.dl, config=config)


    def _make_layer(self, block, planes, num_blocks, stride, iden, dl, config):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList([])
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, iden+str(i), dl, config=config))
            i += 1
            self.in_planes = planes * block.expansion
        #return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        results = OrderedDict()
        print('input shape', x.shape)

        results['input'] = x
        ii = 0
        A, S, l_recon, l_spar, l_inde = self.encoder(x)
        print('encoded A', A.shape)
        print('encoded S', S.shape)
        results['d'+str(ii)] = A
        ii += 1

        results = OrderedDict()
        A = self.conv1(A)
        A = self.bn1(A)
        A = F.relu(A)
        results['d'+str(ii)] = A
        ii += 1
        print('after first conv', A.shape)
        for layer in self.layer1:
            A, rr = layer((A, None))
            results.update(rr)
            print('after first bottleneck', A.shape)
        for layer in self.layer2:
            A, rr = layer((A, rr))
            results.update(rr)
            print('after second bottleneck', A.shape)
        for layer in self.layer3:
            A, rr = layer((A, rr))
            results.update(rr)
            print('after second bottleneck', A.shape)
        for layer in self.layer4:
            A, rr = layer((A, rr))
            results.update(rr)
            print('after third bottleneck', A.shape)

        # A = self.pooll((A, rr))
        # A = self.viewl(A)
        # results['d'+str(ii)] = A
        # ii += 1
        # results['d'+str(ii)] = A
        # ii += 1
        out = self.decoder(A, S)
        # out = self.linear(A)
        return out, results, l_recon+l_spar+l_inde

def ICA_ResNet18(config):
    return ICA_ResNet(BasicBlock, [2, 2, 2, 2], config=config)


def ICA_ResNet34(config):
    return ICA_ResNet(BasicBlock, [3, 4, 6, 3], config=config)


def ICA_ResNet50(config):
    return ICA_ResNet(Bottleneck, [3, 4, 6, 3], config=config)


def ICA_ResNet101(config):
    return ICA_ResNet(Bottleneck, [3, 4, 23, 3], config=config)


def ICA_ResNet152(config):
    return ICA_ResNet(Bottleneck, [3, 8, 36, 3], config=config)




if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    m = ICA_ResNet50()
    o, res, ica_loss = m(x)
