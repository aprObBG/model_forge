import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import *
import random

class Neg_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Neg_Entropy_Loss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b

class PConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=None, config=''):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(PConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
        )


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, iden='na', dl=16):
        super(BasicBlock, self).__init__()
        self.id = iden
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x = x[0]
        results = OrderedDict()
        ii = 0
        out = F.relu(self.bn1(self.conv1(x)))
        results[str(self.id)+str(ii)] = out
        ii += 1
        out = self.bn2(self.conv2(out))
        results[str(self.id)+str(ii)] = out
        ii += 1
        out += self.shortcut(x)
        out = F.relu(out)
        results[str(self.id)+str(ii)] = out
        ii += 1
        return out, results

class BasicBlockQuant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, iden='na', config=''):
        super(BasicBlockQuant, self).__init__()
        self.id = iden
        self.conv1 = Conv2dQuant(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, config=config)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dQuant(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, config=config)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dQuant(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, config=config),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x = x[0]
        results = OrderedDict()
        ii = 0
        out = F.relu(self.bn1(self.conv1(x)))
        results[str(self.id)+str(ii)] = out
        ii += 1
        out = self.bn2(self.conv2(out))
        results[str(self.id)+str(ii)] = out
        ii += 1
        out += self.shortcut(x)
        out = F.relu(out)
        results[str(self.id)+str(ii)] = out
        ii += 1
        return out, results




class Decoder(nn.Module):
    def __init__(self, dl, num_classes=100, config=''):
        super(Decoder, self).__init__()
        self.cls = nn.Sequential(
            BasicBlockQuant(64*3, 256, stride=1, config=config),
            BasicBlockQuant(256, 256, stride=2, config=config),
            BasicBlockQuant(256, 512, stride=1, config=config),
            BasicBlockQuant(512, 512, stride=2, config=config),
            BasicBlockQuant(512, 512, stride=1, config=config),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.lr = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(self.last_channel, num_classes),
            LinearQuant(512, num_classes, config=config),
        )
        self.trans_stride = 1
        self.trans_padding = 0
        self.dl = dl
        self.config = config

    def forward(self, A, S):
        if self.config["quantization"].lower() == "fixed":
            A = quant.fixed_nn(A, self.config["activation_i_width"], self.config["activation_f_width"])
            S = quant.fixed_nn(S, self.config["activation_i_width"], self.config["activation_f_width"])
        if A.shape[-2:] == (1, 1):
            S_shape = S.shape
            A = A.reshape(A.shape[0], -1, self.dl) # Only works when A.shape[-2:] == [1, 1]
            S = S.reshape(S_shape[0], S_shape[1], -1)
            x = (A@S).reshape(S_shape[0], -1, *S_shape[-2:])
        else:
            A = A.reshape(A.shape[0], self.dl, -1, A.shape[1])
            x = torch.empty(A.shape[0], A.shape[1], 3, S.shape[-2], S.shape[-1], dtype=A.dtype).to(A.device)
            for ii in range(A.shape[0]):
                x[ii] = F.conv_transpose2d(A[ii], S[ii], stride=self.trans_stride, padding=self.trans_padding)
            x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        if self.config["quantization"].lower() == "fixed":
            x = quant.fixed_nn(x, self.config["activation_i_width"], self.config["activation_f_width"])
        x, _ = self.cls((x, None))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dl, ica_losses=[0.1, 0.1, 0.1]):
        super(Encoder, self).__init__()
        self.dl = dl
        self.route_AS = nn.Sequential(
            BasicBlock(3, 64, stride=1),
            BasicBlock(64, 64, stride=2),
            BasicBlock(64, 128, stride=1),
        )
        self.route_A = nn.Sequential(
            BasicBlock(64, self.dl, stride=1),
        )
        self.route_S = nn.Sequential(
            BasicBlock(64, 128, stride=1),
            BasicBlock(128, 128, stride=2),
            BasicBlock(128, self.dl*3, stride=1),
        )
        self.trans_padding = 3
        self.trans_stride = 2
        self.recon_loss = nn.MSELoss()
        self.neg_entropy = Neg_Entropy_Loss()
        self.ica_losses = ica_losses

    def forward(self, x):
        AS, _ = self.route_AS((x, None))
        A, S = torch.chunk(AS, 2, 1)
        A, _ = self.route_A((A, None))
        S, _ = self.route_S((S, None))
        b, _, pa, _ = A.shape
        _, _, ps, _ = S.shape
        S = S.reshape(b, self.dl, 3, ps, ps)
        if self.training:
        # if False:
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
            return A, S, self.ica_losses[0]*l1, self.ica_losses[1]*l2, self.ica_losses[2]*l3
        return A, S, 0, 0, 0


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, config, num_classes):
        super(VGG, self).__init__()
        self.dl = 3
        self.features = self._make_layers(cfg[vgg_name], config=config)
        self.viewl = View()
        self.classifier = LinearQuant(512, num_classes, config=config)

        # weight initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        results = OrderedDict()
        ii = 0
        for layer in self.features:
            x = layer(x)
            results['c'+str(ii)] = x
            ii += 1

        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        results['l'+str(ii)] = out
        ii += 1
        return out, results

    def _make_layers(self, cfg, config):
        layers = nn.ModuleList([])
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.extend([nn.AvgPool2d(kernel_size=2, stride=2)])
            else:
                layers.extend([Conv2dQuant(in_channels, x, kernel_size=3, padding=1, config=config),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           QuantLayer(config=config)])
                in_channels = x
        layers.extend([nn.AvgPool2d(kernel_size=1, stride=1)])
        return layers
