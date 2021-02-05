import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import sys
import numpy as np
import numpy.linalg as la
import numpy.random as rd
import scipy.stats
from scipy import spatial

from models import *
from icanet_models import *

import copy, types
from collections import OrderedDict
import config

np.set_printoptions(threshold=10000)
np.set_printoptions(suppress=True)

def count_param(model, learn_layer_dist):
    translation_dict = OrderedDict()
    num_list = []
    total_list = []
    for layer in model.named_parameters():
      numm = len(layer[1].flatten())
      total_list.append(numm)
      layer[1].requires_grad = False
      for learn in learn_layer_dist:
          if (layer[0].startswith(learn)):
            layer[1].requires_grad = True
            layer[1].freezeq = True
            num_list.append(numm)
    return sum(num_list), sum(total_list)


def freeze_backbone(model, learn_layer_dist):
    translation_dict = OrderedDict()
    for layer in model.named_parameters():
      print(layer[0], end = ' ')
      layer[1].requires_grad = False
      for learn in learn_layer_dist:
          if (layer[0].startswith(learn)):
            layer[1].requires_grad = True
            layer[1].freezeq = True
      print(layer[1].requires_grad)
    return 

def freeze_encoder(model):
    model.conv1.conv.weight.requires_grad = False
    model.conv1.conv.bias.requires_grad = False
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

def random_model(model, learn_layer_dist):
    for layer in model.named_parameters():
      for learn in learn_layer_dist:
          if (learn in layer[0]):
            if ('weight' in layer[0]):
                init_layer(layer[1], learn_layer_dist[learn])
            elif ('bias' in layer[0]):
                init_layer(layer[1], 'c')
            else:
                init_layer(layer[1], 'z')
          else:
            if ('weight' in layer[0]):
                init_layer(layer[1], 'n')
            elif ('bias' in layer[0]):
                init_layer(layer[1], 'c')
            else:
                init_layer(layer[1], 'z')


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    tqdm_loader = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(tqdm_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        tqdm_loader.set_description(f'Epoch: {epoch}, loss: {loss.item():.6f}')

def final_test(model, oracle, device, train_loader, test_loader, criterion, nprint=False):
    model.eval()
    oracle.eval()
    test_loss = 0
    correct = 0
    train_correct = 0
    oracle_correct = 0
    correct_vector = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, ica_loss = model(data)
            test_loss += criterion(output, target).item()
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            if (oracle is not None):
                oracle_output, _o, ica_loss_t = oracle(data)
                _, oracle_pred = oracle_output.max(1)  # get the index of the max log-probability
                oracle_correct += pred.eq(oracle_pred).sum().item()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            train_output, _, __ = model(data)
            _, train_pred = train_output.max(1)  # get the index of the max log-probability
            train_correct += train_pred.eq(target).sum().item()


    test_loss /= len(test_loader.dataset)

    if (nprint):
        print('\nTest set: Average loss: {:.4f}, Dataset Accuracy: {}/{} ({:.0f}%) Oracle Accuracy: {}/{} ({:.0f}%) Trainset Accuracy: ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            oracle_correct, len(test_loader.dataset),
            100. * oracle_correct / len(test_loader.dataset),
            100. * train_correct  / len(train_loader.dataset)
            ))


def test(model, oracle, device, test_loader, criterion, nprint=False):
    model.eval()
    oracle.eval()
    test_loss = 0
    correct = 0
    oracle_correct = 0
    correct_vector = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, ica_loss = model(data)
            test_loss += criterion(output, target).item()
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            if (oracle is not None):
                oracle_output, _o, ica_loss_t = oracle(data)
                _, oracle_pred = oracle_output.max(1)  # get the index of the max log-probability
                oracle_correct += pred.eq(oracle_pred).sum().item()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            train_output, _ = model(data)
            _, train_pred = train_output.max(1)  # get the index of the max log-probability
            train_correct += train_pred.eq(target).sum().item()


    test_loss /= len(test_loader.dataset)

    if (nprint):
        print('\nTest set: Average loss: {:.4f}, Dataset Accuracy: {}/{} ({:.0f}%) Oracle Accuracy: {}/{} ({:.0f}%) Trainset Accuracy: ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            oracle_correct, len(test_loader.dataset),
            100. * oracle_correct / len(test_loader.dataset),
            100. * train_correct  / len(train_loader.dataset)
            ))




def noncptest(model, learn_model, oracle, device, test_loader, criterion):
    model.eval()
    oracle.eval()
    test_loss = 0
    correct = 0
    oracle_correct = 0
    correct_vector = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = learn_model((_['352'], None))
            test_loss += criterion(output, target).item()
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            if (oracle is not None):
                oracle_output, _ = oracle(data)
                _, oracle_pred = oracle_output.max(1)  # get the index of the max log-probability
                oracle_correct += pred.eq(oracle_pred).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Dataset Accuracy: {}/{} ({:.0f}%) Oracle Accuracy: {}/{} ({:.0f}%) \n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        oracle_correct, len(test_loader.dataset),
        100. * oracle_correct / len(test_loader.dataset)
        ))

def get_data_loaders():
    test_dataset = datasets.CIFAR10('./', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]), download=False)
    train_dataset = datasets.CIFAR10('./', train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))

    if (args.admode == 'testing'):
        n_samples = int(len(test_dataset)*args.proportion)
        subset1_indices = list(range(0, n_samples))
        subset2_indices = list(range(n_samples, len(test_dataset)))
        adversarial_loader = torch.utils.data.DataLoader(Subset(test_dataset, subset1_indices), batch_size = args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        rtest_loader = torch.utils.data.DataLoader(Subset(test_dataset, subset2_indices), batch_size = args.test_batch_size, shuffle=True, num_workers=1, pin_memory=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    elif (args.admode == 'training'):
        if (args.proportion != 0):
            n_samples = int(len(train_dataset)*args.proportion)
            subset1_indices = list(range(0, n_samples))
            subset2_indices = list(range(n_samples, len(train_dataset)))
            adversarial_loader = torch.utils.data.DataLoader(Subset(train_dataset, subset1_indices), batch_size = args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
            rtest_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=True, num_workers=2, pin_memory=True)
            train_loader = torch.utils.data.DataLoader(
                Subset(train_dataset, subset2_indices),
                batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
        else:
            n_samples = 0
            adversarial_loader = None
            rtest_loader = torch.utils.data.DataLoader(test_dataset, 
                    batch_size = args.test_batch_size, shuffle=True, num_workers=1,
                    pin_memory=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    else:
        n_samples = int(len(test_dataset)*args.proportion)
        subset1_indices = list(range(0, n_samples))
        subset2_indices = list(range(n_samples, len(test_dataset)))
        adversarial_loader = torch.utils.data.DataLoader(Subset(test_dataset, subset1_indices), batch_size = args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        rtest_loader = torch.utils.data.DataLoader(Subset(test_dataset, subset2_indices), batch_size = args.test_batch_size, shuffle=True, num_workers=1, pin_memory=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)


    return adversarial_loader, train_loader, rtest_loader

def print_last(model, learn_layer_dist, ouf):
    for layer in model.named_parameters():
      for learn in learn_layer_dist:
          if (learn in layer[0]):
            print(layer[1].cpu().detach().numpy().copy().flatten().tolist(), file=ouf)

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda:2")
    ad_loader, train_loader, test_loader = get_data_loaders()
    model = ResNet50().to(device)


    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=30, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(model, None, device, test_loader, criterion)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.workspace, "mnist_cnn.pt"))

def calc_dist(model, ref_model):
    for (layer, layer_ref) in zip(model.named_parameters(), ref_model.named_parameters()):
        print(layer[0])
        n_layer = layer[1].cpu().detach().numpy().copy().flatten()
        n_layer = scipy.stats.zscore(n_layer)
        n_layer_ref = layer_ref[1].cpu().detach().numpy().copy().flatten()
        n_layer_ref = scipy.stats.zscore(n_layer_ref)
        nn_norm = np.linalg.norm(n_layer_ref) - np.linalg.norm(n_layer)
        nnn_norm = np.linalg.norm(n_layer_ref-n_layer)
        print("norm diff, diff norm", nn_norm, nnn_norm)
        print("cosine", 1-spatial.distance.cosine(n_layer, n_layer_ref))

def manual_loss(args, criterion, output, target, oracle_pred, mode):
    if (mode == 'r_oracle'):
        simple_loss = criterion(output, oracle_pred)
        return simple_loss
    elif (mode == 'c_oracle'):
        simple_loss = criterion(output, oracle_pred)
        return simple_loss
    elif (mode == 'c_mix'):
        simple_loss = criterion(output, oracle_pred)
        simple_loss3 = criterion(output, target)
        return simple_loss+simple_loss3
    elif (mode == 'c_gt'):
        simple_loss3 = criterion(output, target)
        return simple_loss3


def learn_optimize(args, pre_model, model, oracle, device, train_loader, optimizer, epoch, criterion):
    oracle.eval()
    pre_model.eval()
    model.train()
    tqdm_loader = tqdm(train_loader)
    oracle_correct = 0
    for batch_idx, (data, target) in enumerate(tqdm_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        temp_output, intermediate_results = pre_model(data)

        t_in = intermediate_results['352']
        tt_output = model((t_in, None))
        f_tt_o = tt_output
        f_tt_o = f_tt_o.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred = tt_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        oracle_output, _ = oracle(data)
        oracle_pred = oracle_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        f_o_o = oracle_output
        f_o_o = f_o_o.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
        oracle_correct += pred.eq(oracle_pred.view_as(pred)).sum().item()
        loss = manual_loss(args, criterion, tt_output, target, f_o_o, 'c_oracle')
        loss.backward()
        optimizer.step()
        tqdm_loader.set_description(f'Epoch: {epoch}, loss: {loss.item():.6f}')
    print('Training Oracle Accuracy: {}/{} ({:.0f}%)'.format(
    oracle_correct, len(train_loader.dataset),
    100. * oracle_correct / len(train_loader.dataset)))


def optimize(args, model, oracle, device, train_loader, optimizer, epoch, criterion):
    oracle.eval()
    model.train()
    tqdm_loader = tqdm(train_loader)
    oracle_correct = 0
    for batch_idx, (data, target) in enumerate(tqdm_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _, ica_loss = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        oracle_output, _, ica_loss_or = oracle(data)
        oracle_pred = oracle_output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
        oracle_correct += pred.eq(oracle_pred.view_as(pred)).sum().item()
        cls_loss = criterion(output, target)
        loss = manual_loss(args, criterion, output, target, oracle_pred, 'c_oracle')
        loss.backward()
        optimizer.step()
        tqdm_loader.set_description(f'Epoch: {epoch}, loss: {loss.item():.6f}')
    print('Training Oracle Accuracy: {}/{} ({:.0f}%)'.format(
    oracle_correct, len(train_loader.dataset),
    100. * oracle_correct / len(train_loader.dataset)), file=sys.stderr)

def weak_create(model, learn_layer_dist):
    module_list = list(model.children())
    num_list = []

    i = 0
    for layer in model.named_parameters():
        for learn in learn_layer_dist:
            if (layer[0].startswith(learn)):
                num_list.append(i)
        i += 1

    pre_model_list = nn.ModuleList([])
    model_list = nn.ModuleList([])
    post_model_list = nn.ModuleList([])
    hold_model_list = nn.ModuleList([])
    i = 0
    pre_flag = True
    post_flag = False
    for layer in module_list:
        if isinstance(layer, nn.modules.container.ModuleList):
            for secondlayer in layer:
                set_flag = False
                for thirdlayer in secondlayer.children():
                    for forth in thirdlayer.named_parameters():
                        if (i in num_list):
                            set_flag = True
                        i += 1
                if (set_flag):
                    curr = copy.deepcopy(secondlayer)
                    model_list.append(curr)
                    pre_flag = False
                    post_flag = False
                elif (not pre_flag):
                    post_flag = True

                if (pre_flag):
                    curr = copy.deepcopy(secondlayer)
                    pre_model_list.append(curr)
                if (post_flag):
                    curr = copy.deepcopy(secondlayer)
                    post_model_list.append(curr)


        else:
            set_flag = False
            for forth in layer.named_parameters():
                if (i in num_list):
                    set_flag = True
                i += 1
            if (set_flag):
                curr = copy.deepcopy(layer)
                model_list.extend(hold_model_list)
                model_list.append(curr)
                hold_model_list = nn.ModuleList([])
                pre_flag = False
            elif (not pre_flag):
                curr = copy.deepcopy(layer)
                hold_model_list.append(curr)
                post_flag = True

            if (pre_flag):
                curr = copy.deepcopy(layer)
                pre_model_list.append(curr)
            if (post_flag):
                curr = copy.deepcopy(layer)
                post_model_list.append(curr)

    created_model = nn.Sequential(*model_list)
    pre_model = nn.Sequential(*pre_model_list)
    post_model = nn.Sequential(*post_model_list)
    return created_model, pre_model, post_model



def set_layer_in_model(model, learn_model, learn_layer_dist):
    i = 0
    learn_model_list = []

    for layer in learn_model.named_parameters():
        learn_model_list.append(layer)

    for layer in model.named_parameters():
        set_flag = False
        for learn in learn_layer_dist:
            if (layer[0].startswith(learn)):
                set_flag = True
        if (set_flag):
            if True:
                print('from', learn_model_list[i][0], learn_model_list[i][1].shape, 'set', layer[0], layer[1].shape)
            with torch.no_grad():
              for tensor, target_tensor in zip(layer[1], learn_model_list[i][1]):
                  tensor.copy_(target_tensor.detach().clone())
            i += 1
    return None

def ica_net_train(args):
    config_file = 'config.yaml'
    model_type = 'UNET'
    configk = config.Configuration(model_type, config_file)
    print(configk.get_config_str(), file=sys.stderr)
    configk = configk.config_dict

    torch.manual_seed(args.seed)
    device = torch.device("cuda:2")
    ad_loader, train_loader, test_loader = get_data_loaders()
    model = ICA_ResNet50(configk).to(device)
    simp_net = torch.load(os.path.join(args.workspace, "ckpt.pth"))
    model.load_state_dict(simp_net['net'])


    if False:
        for layer in model.named_parameters():
            print(layer[0])
    if False:
        for layer in model.children():
            print(layer)
    configkk = copy.deepcopy(configk)
    configkk['quantization']='FIXED'
    configkk['activation_f_width']=8
    configkk['activation_i_width']=8
    configkk['weight_f_width']=8
    configkk['weight_i_width']=8
    model_save = ICA_ResNet50(configk).to(device)
    model_save.load_state_dict(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    r_criterion = nn.CrossEntropyLoss()

    ll = 0
    dist_list = []
    t_layer_dist = {}


    ll = 0
    dist_list = []
    t_layer_dist = {}
    encoder_dist = {}
    decoder_dist = {}
    key_save = []
    ff = False
    for layer in reversed(list(model.named_parameters())):
      ff = False
      if 'linear' in layer[0]: 
        t_layer_dist[layer[0]] = 'c'
      elif 'encoder' in layer[0]:
        encoder_dist[layer[0]] = 'n'
        ff = True
      else:
        t_layer_dist[layer[0]] = 'n'
      ll += 1
      if ('weight' in layer[0]) and not ff:
          dist_list.append(copy.deepcopy(t_layer_dist))
    total_length = ll

    i = 0
    for dist in dist_list:
        learn_layer_dist = dist

        curr_param, t_param = count_param(model, learn_layer_dist)
        print(str(len(learn_layer_dist))+'/'+str(total_length), end=' ')
        print(str(curr_param)+'/'+str(t_param), )

        learn_model, pre_learn_model, post_learn_model = weak_create(model, learn_layer_dist)
        learn_model_save, pre_learn_model_save, post_learn_model_save = weak_create(model_save, learn_layer_dist)

        random_last_layer(model, learn_layer_dist)

        r_lr = 1e-3

        if (False):
            test(model, model_save, device, test_loader, criterion, nprint=True)
        if (False):
            for layer in model.named_parameters():
              print(layer[0], end = ' ')
              print(layer[1].requires_grad)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        learn_optimizer = optim.Adam(learn_model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
        learn_scheduler = StepLR(learn_optimizer, step_size=20, gamma=args.gamma)
        switch = args.switch
        for epoch in range(1, 60 + 1):
            if (switch == 'direct'):
                optimize(args, model, model_save, device, ad_loader, optimizer, epoch, criterion)
            elif (switch == 'extract'):
                learn_optimize(args, model, learn_model, model_save,
                        device, ad_loader, learn_optimizer, epoch, r_criterion)
                set_layer_in_model(model, learn_model, learn_layer_dist)
                noncptest(model, learn_model, model_save, device, test_loader, criterion)
            else:
                print('no switch')
                exit()

            learn_scheduler.step()
            scheduler.step()
        final_test(model, model_save, device, ad_loader, test_loader, criterion, nprint=True)

    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--proportion', type=float, default=0.5, metavar='M',
                        help='Size of the adversarial set with respect to test set (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--admode', type=str, default='test', metavar='N',
                        help='where the adversarial examples come from (training/testing)')
    parser.add_argument('--switch', type=str, default='direct', metavar='N',
                        help='whole model or partial adversarial training (direct/extracted)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=bool, default=True,
                        help='For Saving the current Model')
    parser.add_argument('--train-freeze', action='store_true', default=False,
                        help='Train base model, or train enocder+backbone freezed model.')
    args = parser.parse_args()
    args.workspace = ''

    if not args.train_freeze:
        print(f'******************Start training base model***************')
        main(args)
    else:
        print(f'******************Start training freezed model***************', file=sys.stderr)
        ica_net_train(args)

