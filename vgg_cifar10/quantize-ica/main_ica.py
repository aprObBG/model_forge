'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import config
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--data', default='', type=str)
parser.add_argument('--set', choices=['cifar10', 'cifar100'], default='cifar10', type=str)
parser.add_argument('--arch', default='vgg19', type=str)
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('-b', '--batch', default=512, type=int)
parser.add_argument('--optim', default='sgd', type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

torch.manual_seed(args.seed)

config_file = 'config.yaml'
model_type = 'UNET'
config = config.Configuration(model_type, config_file)
print(config.get_config_str())
config = config.config_dict


device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

mean_std = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean_std[args.set][0], mean_std[args.set][1]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_std[args.set][0], mean_std[args.set][1]),
])

if args.set == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root=args.data, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=args.data, train=False, download=False, transform=transform_test)
else:
    trainset = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
assert args.arch.lower() in ['vgg19', 'vgg16', 'vgg13', 'vgg11']
num_classes = 10 if args.set == 'cifar10' else 100
net = VGG(args.arch.upper(), config=config, num_classes=num_classes)
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
    checkpoint = torch.load(f'./checkpoint/{args.set}_{args.arch.upper()}_{args.optim}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# learn_dict = {'encoder': 'n'}
# freeze_encoder(net, learn_dict)
# random_last_layer(net, learn_dict)


criterion = nn.CrossEntropyLoss()
if args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    raise NotImplementedError

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300], gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, threshold=1e-5, min_lr=1e-8, verbose=True)


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
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
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
            outputs, _ = net(inputs)
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
        torch.save(state, f'./checkpoint/{args.set}_{args.arch.upper()}_{args.optim}.pth')
        best_acc = acc

    return test_loss

for epoch in range(start_epoch, 300):
    train(epoch)
    test_loss = test(epoch)
    scheduler.step(test_loss)
