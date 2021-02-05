#! /usr/bin/env python3
# -*- coding: utf-8 -**-

import ast
import math
import collections
import sys

def vgg():
  # arch = collections.OrderedDict([
  #     ("convi",{'in_channel': 3, 'in_height': 32, 'in_width': 32, 'fil_height': 3, 'fil_width': 3, 'out_channel': 64,  'in_quant': 8, 'fil_quant': 8, 'padding':'same'}),
  #     ("relui",{'in_channel':64, 'in_height': 32, 'in_width': 32, 'quant': 32})
  #   ])
  arch = collections.OrderedDict([])

  j_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
  init_dim = 32
  init_chan = 3
  pool_fac = 1
  in_chan = init_chan
  for i in range(len(j_list)):
    if (j_list[i] == 'M'):
      pool_fac *= 2
    else:
      key = 'layer'+str(i)+'.'+'.'+'conv1'
      line = {'in_channel': in_chan, 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'fil_height': 3, 'fil_width': 3, 'out_channel': j_list[i], 'in_quant': 16, 
          'fil_quant': 16, 'padding':'same'}
      arch[key] = line

      key = 'layer'+str(i)+'.'+'.'+'relu1'
      line = {'in_channel': j_list[i], 'in_height': init_dim/pool_fac, 'in_width': init_dim/pool_fac, 
          'quant': 32}
      arch[key] = line

      in_chan = j_list[i]

  arch["fc1"] = {'in_height': 512, 'in_width': 10, 'fil_height': 10, 'fil_width': 512, 'in_quant': 16, 'fil_quant': 16}

  # for key in arch:
  #   print(key)
  #   print(arch[key])
  # exit()

  return arch
