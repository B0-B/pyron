#!/usr/bin/python3
import sys
sys.path.append('./..')
import os
from PIL import Image
import numpy as np
from pyron import CNN, CrossEntropyLoss, fuse


nn = CNN()
nn.sequential('Number Detector', 
       topology=[5, 3, 3, 1],
       activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])
nn.initialize('uniform',
              weight_range=[-.2, .2],
              bias_range=[-.2, .2])

fuse(nn, np.float32)
