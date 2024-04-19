#!/usr/bin/python3
import sys
sys.path.append('./..')
import os
from PIL import Image
import numpy as np
from pyron import CNN, CrossEntropyLoss, FusedNetwork


def testsuite () -> bool:

       full_size_model = CNN()
       full_size_model.sequential('Number Detector', 
              topology=[5, 3, 3, 1],
              activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])
       full_size_model.initialize('uniform',
                     weight_range=[-.2, .2],
                     bias_range=[-.2, .2])

       # fuse the full size model
       # fused_model = fuse(full_size_model, np.float32)
       fused_model = FusedNetwork.load_from_model(full_size_model)

       result = fused_model.forward(np.random.uniform(-1, 1, (5,)))
       print('fused result', result)

       if len(result.shape) == 1 and result.shape[0] == 1:
              return True
       return False

if __name__ == '__main__':

       print('test result:', testsuite())