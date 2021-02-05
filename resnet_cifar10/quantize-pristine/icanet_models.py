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

    def __init__(self, in_planes, planes, stride=1, iden='na', dl=16):
        super(Bottleneck, self).__init__()

        self.id = iden

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, groups=dl)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(dl, planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=dl)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(dl, planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, groups=dl)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.bn3 = nn.GroupNorm(dl, self.expansion*planes)
        self.relu3 = nn.ReLU()

        #self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            self.shortcutconv1 = nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, groups=dl)
            # self.shortcutbn1 = nn.BatchNorm2d(self.expansion*planes)
            self.shortcutbn1 = nn.GroupNorm(dl, self.expansion*planes)

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

class Neg_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Neg_Entropy_Loss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b

class Decoder(nn.Module):
    def __init__(self, dl, num_classes=10):
        super(Decoder, self).__init__()
        self.cls = nn.Sequential(
            ConvBNCELU(384, 128, kernel_size=3, padding=1, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lr = nn.Linear(128, num_classes)
        self.trans_stride = 2
        self.trans_padding = 1
        self.dl = dl

    def forward(self, A, S):
        A = A.reshape(A.shape[0], -1, self.dl, *A.shape[-2:])
        x = torch.empty(A.shape[0], A.shape[1], 3, *S.shape[-2:], dtype=A.dtype).to(A.device)
        for ii in range(A.shape[0]):
            x[ii] = F.conv_transpose2d(A[ii], S[ii], stride=self.trans_stride, padding=self.trans_padding)
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        x = self.cls(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.lr(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dl):
        super(Encoder, self).__init__()
        self.dl = dl
        self.route_AS = nn.Sequential(
            ConvBNCELU(3, 16, kernel_size=3, padding=1, stride=1),
            ConvBNCELU(16, 32, kernel_size=3, padding=1, stride=2),
        )
        self.route_A = nn.Sequential(
            ConvBNCELU(16, self.dl, kernel_size=1, padding=0, stride=1),
        )
        self.route_S = nn.Sequential(
            ConvBNCELU(16, self.dl*3, kernel_size=3, padding=1, stride=2),
        )
        self.trans_padding = 3
        self.trans_stride = 2
        self.recon_refine = ConvBNCELU(self.dl, 3, kernel_size=1)
        self.recon_loss = nn.MSELoss()
        self.neg_entropy = Neg_Entropy_Loss()

    def forward(self, x):
        AS = self.route_AS(x)
        A, S = torch.chunk(AS, 2, 1)
        A = self.route_A(A)
        S = self.route_S(S)
        b, _, pa, _ = A.shape
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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ICA_ResNet, self).__init__()
        self.dl = 16
        self.in_planes = 64
        self.conv1 = nn.Conv2d(self.dl, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, iden='1', dl=self.dl)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, iden='2', dl=self.dl)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, iden='3', dl=self.dl)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, iden='4', dl=self.dl)
        self.pooll = Pooll()
        self.viewl = View()
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        self.encoder = Encoder(dl=self.dl)
        self.decoder = Decoder(dl=self.dl)


    def _make_layer(self, block, planes, num_blocks, stride, iden, dl):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList([])
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, iden+str(i), dl))
            i += 1
            self.in_planes = planes * block.expansion
        #return nn.Sequential(*layers)
        return layers

    def forward(self, x):

        A, S, l_recon, l_spar, l_inde = self.encoder(x)

        ii = 0
        results = OrderedDict()
        A = self.conv1(A)
        A = self.bn1(A)
        A = F.relu(A)
        results['d'+str(ii)] = A
        ii += 1
        for layer in self.layer1:
            A, rr = layer((A, None))
            results.update(rr)
        for layer in self.layer2:
            A, rr = layer((A, rr))
            results.update(rr)
        for layer in self.layer3:
            A, rr = layer((A, rr))
            results.update(rr)
        for layer in self.layer4:
            A, rr = layer((A, rr))
            results.update(rr)
        # A = self.pooll((A, rr))
        # A = self.viewl(A)
        # results['d'+str(ii)] = A
        # ii += 1
        # A = self.linear(A)
        # results['d'+str(ii)] = A
        # ii += 1
        out = self.decoder(A, S)
        return out, results, l_recon+l_spar+l_inde

def ICA_ResNet18():
    return ICA_ResNet(BasicBlock, [2, 2, 2, 2])


def ICA_ResNet34():
    return ICA_ResNet(BasicBlock, [3, 4, 6, 3])


def ICA_ResNet50():
    return ICA_ResNet(Bottleneck, [3, 4, 6, 3])


def ICA_ResNet101():
    return ICA_ResNet(Bottleneck, [3, 4, 23, 3])


def ICA_ResNet152():
    return ICA_ResNet(Bottleneck, [3, 8, 36, 3])




if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    m = ICA_ResNet50()
    o, res, ica_loss = m(x)
