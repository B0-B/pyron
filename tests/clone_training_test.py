#!/usr/bin/python3
import sys
sys.path.append('./..')
import os, time, numpy as np
from pathlib import Path
from PIL import Image
from pyron import CNN, CrossEntropyLoss, FusedNetwork, Loader, clone_training


def testsuite() -> bool:

    # locate digit dataset of 20 x 20 px images
    datasetPath = Path(__file__).resolve().parent.joinpath('numbersDataset/')
    dataset = Loader.load_image_data(datasetPath, suffix='.png')

    split = 0.8
    split_index = int(split*len(dataset))
    training_set = dataset[:split_index]
    test_set = dataset[split_index:]

    # instantiate the master model
    # & define the topology
    master = CNN()
    master.sequential('Number Detector', 
        topology=[400, 256, 256, 100, 10],
        activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])

    # define hyperparameters for clone training
    hyperparameters = {
        'clone_number': 4,
        'batch_size': 3,
        'epochs': 100,
        'learning_rate': 0.01,
        'split': 0.8,
        'init_method': 'random', 
        'weight_range': [-.1, .1], 
        'bias_range': [-.1, .1],
        # 'mean_weight': 0, # for normal init
        # 'var_weight': 0, 
        # 'mean_bias': 0, 
        # 'var_bias': 0
    }

    # start the training
    merged_model = clone_training(master, training_set, **hyperparameters)

    merged_accuracy = merged_model.test(training_set, mode='argmax')
    print(f'[testing]: merged model categorization accuracy: {merged_accuracy}%')

    return True

if __name__ == '__main__':

    testsuite()