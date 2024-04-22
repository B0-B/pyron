#!/usr/bin/python3
import sys
sys.path.append('./..')
import os
from PIL import Image
import numpy as np
from pyron import CNN, CrossEntropyLoss, Loader
from pathlib import Path

# locate a data set of 20 x 20 px images
datasetPath = Path(__file__).resolve().parent.joinpath('hwdd-20/dataset/')

# load image dataset
dataset = Loader.load_image_data(datasetPath, suffix='.png')


nn = CNN()

nn.sequential('Number Detector', 
       topology=[400, 256, 256, 100, 10],
       activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])

# nn.initialize('normal',
#               mean_bias=0,
#               mean_weight=0,
#               var_bias=0.013,
#               var_weight=0.013)

nn.initialize('uniform',
              weight_range=[-.3, .3],
              bias_range=[-.3, .3])

criterion = CrossEntropyLoss(one_hot_encoded=True, temperature=1)

nn.fit( 
    dataset, 
    error_plot=1, 
    learning_rate=.03, 
    verbose=1, 
    epochs=100, 
    batch_size=10,
    evaluation_criterion=criterion,
    # stop=1., 
    split=0.9
)