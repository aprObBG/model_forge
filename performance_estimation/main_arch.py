#! /usr/bin/env python3
# -*- coding: utf-8 -**-

import ast
import math
import collections
import subprocess
from Crypto.Util import number
import sys
from security_estimate import *
from resnet import resnet
from icaresnet import icaresnet
from vgg import vgg
from icavgg import icavgg
from unet import unet

def main():
  #arch = resnet()
  #arch = icaresnet()
  arch = vgg()

  score = estimate(arch, 'maximum')

  return score

if __name__ == '__main__' :
  main()

