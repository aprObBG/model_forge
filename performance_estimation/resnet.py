#! /usr/bin/env python3
# -*- coding: utf-8 -**-

import ast
import math
import collections
import sys

def resnet():
  in_dim = 3
  resnet_arch = collections.OrderedDict([
      ("convi",{'in_channel': 3, 'in_height': 32, 'in_width': 32, 'fil_height': 3, 'fil_width': 3, 'out_channel': 64,  'in_quant': 8, 'fil_quant': 8, 'padding':'same'}),
      ("relui",{'in_channel':64, 'in_height': 32, 'in_width': 32, 'quant': 32})
    ])
  j_list = [3, 4, 6, 3]

  init_dim = 32
  init_chan = 64
  exp_chan_in = init_chan
  exp_chan_out = init_chan*4
  pool_fac = 1
  for i in range(4):
    for j in range(j_list[i]):
      key = 'layer'+str(i)+'.'+str(j)+'.'+'conv1'
      line = {'in_channel': exp_chan_in, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'fil_height': 1, 'fil_width': 1, 'out_channel': init_chan*pool_fac, 'in_quant': 16, 
          'fil_quant': 16, 'padding':'same'}
      resnet_arch[key] = line

      key = 'layer'+str(i)+'.'+str(j)+'.'+'relu1'
      line = {'in_channel': init_chan*pool_fac, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'quant': 32}
      resnet_arch[key] = line


      key = 'layer'+str(i)+'.'+str(j)+'.'+'conv2'
      line = {'in_channel': init_chan*pool_fac, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'fil_height': 3, 'fil_width': 3, 'out_channel': init_chan*pool_fac, 'in_quant': 16, 
          'fil_quant': 16, 'padding':'same'}
      resnet_arch[key] = line

      key = 'layer'+str(i)+'.'+str(j)+'.'+'relu2'
      line = {'in_channel': init_chan*pool_fac, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'quant': 32}
      resnet_arch[key] = line

      key = 'layer'+str(i)+'.'+str(j)+'.'+'conv3'
      line = {'in_channel': init_chan*pool_fac, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'fil_height': 1, 'fil_width': 1, 'out_channel': exp_chan_out, 'in_quant': 16, 
          'fil_quant': 16, 'padding':'same'}
      resnet_arch[key] = line

      key = 'layer'+str(i)+'.'+str(j)+'.'+'relu3'
      line = {'in_channel': exp_chan_out, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'quant': 32}
      resnet_arch[key] = line


    pool_fac *= 2
    exp_chan_in = init_chan*pool_fac*4
    exp_chan_out = init_chan*pool_fac*4

  resnet_arch["fc1"] = {'in_height': 1024, 'in_width': 10, 'fil_height': 10, 'fil_width': 1024, 'in_quant': 16, 'fil_quant': 16}

  return resnet_arch
