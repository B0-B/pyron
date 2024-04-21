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

    # instantiate the master model
    # & define the topology
    master = CNN()
    master.sequential('Number Detector', 
        topology=[400, 256, 256, 100, 10],
        activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])

    # define hyperparameters
    hyperparameters = {
        'clone_number': 4,
        'batch_size': 10,
        'epochs': 100,
        'learning_rate': 0.05,
        'split': 0.8
    }

    trained_model = clone_training(master, dataset, **hyperparameters)

    return True

if __name__ == '__main__':

    testsuite()