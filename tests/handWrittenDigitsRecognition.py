#!/usr/bin/python3
import sys
sys.path.append('./..')
import os
from PIL import Image
import numpy as np
from pyron import CNN, CrossEntropyLoss
from pathlib import Path

# a data set of 20 x 20 px images
datasetPath = Path(__file__).resolve().parent.joinpath('numbersDataset/')

# load dataset
dataset = []

for file in os.listdir(datasetPath):

    if file.endswith(".png"):

        filePath = os.path.join(datasetPath, file)
        im_frame = Image.open(filePath)

        # pixel array, each value is in [0,100]
        # normalize 0-100 to 0-1 (Min Max Scaling).
        x_vector = np.array(im_frame.getdata()) / 100

        # label or "y" value
        label = int(filePath.split('\\')[-1].split('_')[1])
        y_vector = np.zeros((10,))
        y_vector[label] = 1.

        dataset.append([x_vector, y_vector])


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