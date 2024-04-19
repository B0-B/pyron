#!/usr/bin/python3

'''
Tests if a model is correctly exported and reconstructed after import.
'''


import sys
sys.path.append('./..')
from pyron import CNN
import numpy as np
from traceback import print_exc
from pathlib import Path

def testsuite () -> bool:

    try:

        path = Path('./test_model.npz')

        nn = CNN()

        nn.sequential('test_model_name', 
            topology=[400, 256, 256, 100, 10],
            activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])

        nn.initialize()

        # generate zero input vector and denote output
        input_vec = np.zeros(shape=(nn.topology[0],))
        output = nn.propagate(input_vec)

        nn.save(path)

        clone = CNN()
        clone.load(path)

    except:

        print_exc()

    finally:

        # clean
        path.unlink()

    # confirm propagation
    output_reconstructed = clone.propagate(input_vec)
    if np.all(output == output_reconstructed):
        return True
    
    # otherwise reject
    return False


if __name__ == '__main__':

    print('test result:', testsuite())