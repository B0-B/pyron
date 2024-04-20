#!/usr/bin/python3
import sys
sys.path.append('./..')
import os, time, numpy as np
from PIL import Image
from pyron import CNN, CrossEntropyLoss, FusedNetwork

def testsuite () -> bool:

    sample_size = 10000 # number of iterations

    # create new model
    full_size_model = CNN()
    full_size_model.sequential('Number Detector', 
          topology=[500, 300, 300, 10],
          activation_overlay=['linear', 'relu', 'relu', 'relu', 'linear'])
    full_size_model.initialize('uniform',
                 weight_range=[-.2, .2],
                 bias_range=[-.2, .2])

    # fuse the full size model
    fused_model = FusedNetwork.load_from_model(full_size_model, np.float16)

    dt_full_size, dt_fused = 0, 0

    # test the full size model throughput time
    for i in range(sample_size):

        # sample a random input vector
        sample_input = np.random.uniform(-1, 1, (full_size_model.topology[0],))

        t0 = time.time_ns()
        full_size_model.propagate(sample_input)

        # benchmark
        dt_full_size += time.time_ns() - t0

        # test the full size model throughput time
        t0 = time.time_ns()
        fused_model.forward(sample_input)

        # benchmark
        dt_fused += time.time_ns() - t0
    
    # normalize
    dt_full_size /= sample_size
    dt_fused /= sample_size

    # convert to ms
    dt_full_size /= 1e3 
    dt_fused /= 1e3 

    sample_input = np.random.uniform(-1, 1, (full_size_model.topology[0],))

    sample_output_full = full_size_model.propagate(sample_input)
    sample_output_fuse = fused_model.forward(sample_input)
    
    print(f'Full size model - avg. iteration time ({sample_size} samples): {dt_full_size} ms')
    print(f'Fused model - avg. iteration time ({sample_size} samples): {dt_fused} ms ({round(100*(1 - dt_fused/dt_full_size), 2)}% improvement)')

    print(f'Size of full size model: {full_size_model.size()} bytes')
    print(f'Size of fused size model: {fused_model.size()} bytes')

    print(f'Prop. Output of Full Model:', sample_output_full)
    print(f'Prop. Output of Fused Model:', sample_output_fuse)

    return True

if __name__ == '__main__':

    print('test result:', testsuite())