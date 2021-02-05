'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Subset

import config
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from icanet_models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data', default='', type=str)
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('-b', '--batch', default=128, type=int)
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 3)')

args = parser.parse_args()

torch.manual_seed(args.seed)

config_file = 'config.yaml'
model_type = 'UNET'
config = config.Configuration(model_type, config_file)
print(config.get_config_str())
config = config.config_dict


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data, train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=args.batch, shuffle=True, num_workers=4)

n_samples = int(len(trainset)*1)
subset1_indices = list(range(0, n_samples))
trainloader = torch.utils.data.DataLoader(
    Subset(trainset, subset1_indices),
    batch_size=args.batch, shuffle=True, num_workers=3, pin_memory=True)


testset = torchvision.datasets.CIFAR10(
    root=args.data, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet50()
net = ICA_ResNet50(config)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

def freeze_encoder(model, learn_layer_dist):
    translation_dict = OrderedDict()
    for layer in model.named_parameters():
      print(layer[0], end = ' ')
      layer[1].requires_grad = True
      for learn in learn_layer_dist:
          if (layer[0].startswith(learn)):
            #translation_dict[layer[0]] = f_layer[0]
            layer[1].requires_grad = False
            layer[1].freezeq = False
      print(layer[1].requires_grad)
    return 


def freeze_backbone(model, learn_layer_dist):
    translation_dict = OrderedDict()
    for layer in model.named_parameters():
      print(layer[0], end = ' ')
      layer[1].requires_grad = False
      for learn in learn_layer_dist:
          if (layer[0].startswith(learn)):
            #translation_dict[layer[0]] = f_layer[0]
            layer[1].requires_grad = True
            layer[1].freezeq = True
      print(layer[1].requires_grad)
    return 


def init_layer(layer_param, method):
    if (method == 'n'):
        try:
            torch.nn.init.kaiming_normal_(layer_param)
        except Exception:
            torch.nn.init.normal_(layer_param)
    elif (method == 'c'):
        torch.nn.init.constant_(layer_param, 0.005)
    else:
        torch.nn.init.constant_(layer_param, 0)


def random_last_layer(model, learn_layer_dist):
    for layer in model.named_parameters():
      for learn in learn_layer_dist:
          if (learn in layer[0]):
            if ('weight' in layer[0]):
                init_layer(layer[1], learn_layer_dist[learn])
            elif ('bias' in layer[0]):
                init_layer(layer[1], 'c')
            else:
                init_layer(layer[1], 'z')
    return


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    #learn_layer_dist = {'decoder': 'n'}
    #freeze_backbone(net, learn_layer_dist)
    #random_last_layer(net, learn_layer_dist)

# freeze things
# learn_dict = {'encoder': 'n'}
# freeze_encoder(net, learn_dict)
# random_last_layer(net, learn_dict)


criterion = nn.CrossEntropyLoss()
if args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        outputs, _, ica_loss = net(inputs)
        # print(prof)
        # exit()
        cls_loss = criterion(outputs, targets)
        loss = ica_loss + cls_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, ica_loss = net(inputs)
            exit()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# def adjust_learning_rate(optimizer, epoch, args):
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
test(0)
exit()

for epoch in range(start_epoch, 350):
    train(epoch)
    test(epoch)
    scheduler.step()
