#!/usr/bin/python3

import os, sys, gc

# scientific libraries
import numpy as np
import matplotlib.pyplot as plt
from math import isnan

import copy

from traceback import print_exc
from threading import Thread, Event
from time import sleep
from typing import Callable
from pathlib import Path
from PIL import Image

class thread (Thread):

    '''
    [MODULE]

    The activity of this thread is bound to the termination state of the controller.
    '''

    def __init__ (self, function: Callable, *args, **kwargs) -> None:

        Thread.__init__(self)
        self.func = function
        self.args = args
        self.kwargs = kwargs
        self.stoprequest = Event()
        self.finished = False

    def run (self) -> None:

        self.func(*self.args, **self.kwargs)
        self.finished = True

    def stop (self, timeout = None):

        self.stoprequest.set()
        super(thread, self).join(timeout)

# Class Objects & Types
# ======================================== 
class Criterion:
    pass

class Layer:

    def __init__(self) -> None:
        pass

class Module:

    def __init__(self) -> None:
        pass

class Network:

    def __init__(self) -> None:
        pass


# Loss / Target Functions
# =========================================
# Every loss function is a criterion and has to provide an evaluate method
# evaluate(self, xarray:np.ndarray, yarray:np.ndarray, derivative: bool=False)
# which allows to compute the root and derivative value (w.r.t. x).
class CrossEntropyLoss (Criterion):

    '''
    Categorical Cross Entropy loss or "log" loss 
    - equivalent to Kullback-Leibler Divergence but without logarithmic difference.

    Should be used if activations are held linear (or no activation).
    The raw identical layer input will be activated with a boltzmann-like softmax weighting
    to obtain probabilities. The default format for input and targets are one-hot encoded vectors
    i.e. input=[0.8, 0.15, 0.05], target=[1, 0, 0] 
    
    xarray      The prediction vector or batch tensor - linear activated.
    weights     A 1D ndarray with weights, if none provided 
                will assume uniform distribution
    '''

    def __init__(self, one_hot_encoded:bool=True, derivative: bool=False, weights:None|np.ndarray=None, base:int=np.e, temperature:int=1) -> None:

        self.one_hot_encoded = one_hot_encoded
        self.derivative = derivative  
        self.weights = weights
        self.base = base
        self.beta = 1 / temperature
    
    def evaluate(self, xarray:np.ndarray, yarray:np.ndarray, derivative: bool=False):

        # activate linear prediction with softmax
        boltzman_factor = np.exp( self.beta * xarray )
        xarray = (boltzman_factor.T / np.sum(boltzman_factor, axis=1)).T

        # convert single inputs to batch shape
        if len(yarray.shape) == 1 and len(xarray.shape) == 1:
            xarray = np.reshape(xarray, (1, xarray.shape[0]))
            yarray = np.reshape(yarray, (1, yarray.shape[0]))

        # Clip values to avoid division by zero / log of zero
        epsilon = 1e-8
        xarray = np.clip(xarray, epsilon, 1. - epsilon)
                
        if derivative:
            # placeholder
            return (xarray - yarray)
        
        p = yarray
        log_q = np.emath.logn(self.base, xarray)

        # check if not one-hot encoded
        if not self.one_hot_encoded:
            # create uniform weight vector
            if not self.weights:
                self.weights = np.ones(yarray[0].shape) / yarray.shape[0]
            weighted_p = np.array([self.weights[y[0]] * xarray[i, y[0]] for i, y in enumerate(p)])
            # return - ( weighted_p * log_q - (1-weighted_p) * (1-log_q) ).sum(1).mean()
            return - ( weighted_p * log_q ).sum(1).mean()
        
        # otherwise treat the xarray as hot-ncoded inputs
        # this allows direct element-wise multiplication of samples,
        # we sum up all sample vectors to scalars, and mean accross the batch.
        else:
            return - ( log_q * yarray ).sum(1).mean()

class MeanSquaredError (Criterion):

    '''
    MSE - Mean Squared Error
    Good for regression tasks.
    '''

    def __init__(self) -> None:
        pass
        
    def evaluate (self, xarray:np.ndarray, yarray:np.ndarray, derivative: bool=False):

        delta = xarray - yarray
        if derivative: return 2 * delta
        return  ( delta ** 2 ).mean()



# Convolutional Neural Network (Perceptron)
# =========================================
class CNN (Network):

    '''
    Convolutional Neural Network (Perceptron)

    [Call Tree]
    propagate
      └ propagate_to_layer
          └ transpose
              └ activate(tensor.dot(layer) + bias)
    '''

    def __init__ (self):

        super().__init__()
        self.initialized: bool=False

        # general information
        self.fit_label = {}
        self.name = '',

        # geometry and specs
        self.activation_overlay = []
        self.L = 0
        self.topology = []
        self.parameters = 0
        self.float_precision = np.float64

        # hyper parameters
        self.learning_rate = 0.01
        self.learning_decay = 0.00
        self.momentum = 0.05        

        # model's main storage
        self.tensors = {}
        self.last_weight_change = {}
        self.weight_gradients = {}
        self.bias_gradients = {}
        self.biases = {}
        
        self.default_activation = 'sigmoid'

        self.accuracy = None
        
    def sequential (self, name: str='new_network', topology: list=[], activation_overlay: list[str]|None=None): # works

        '''
        Will setup a new sequential architecture.
        The setup topology looks like

        layer 0         1(hidden)   2        L-1        L
            input   >   °     >     °         °     >   Out
            input   >   °     >     °   ...   °     >   Out
            input   >   °     >     °         °     >   Out
            input   >   °     >     °         °     >   Out
        '''

        self.topology = topology
        self.L = len(self.topology) - 1
        self.name = name

        # apply activation overlay
        if activation_overlay:

            self.activation_overlay = ['sigmoid' for i in range(self.L+1)]
            self.activation_overlay[0] = 'linear'
            self.activation_overlay[-1] = 'linear'
        
        else:

            if len(activation_overlay)-1 != self.L:

                raise ValueError('the provided activation overlay must have same dimension as the total layer number.')

            self.activation_overlay = activation_overlay

        for layer in range(1, self.L + 1):

            # generate only L-1 transfer matrices
            self.tensors[layer] = np.zeros( shape=( self.topology[layer], self.topology[layer-1] ), dtype=self.float_precision )

            # a second skeleton to cache changes
            self.last_weight_change[layer] = np.zeros( shape=( self.topology[layer], self.topology[layer-1] ), dtype=self.float_precision )
            self.weight_gradients[layer] = np.zeros( shape=( self.topology[layer], self.topology[layer-1] ), dtype=self.float_precision )
            self.bias_gradients[layer] = np.zeros( shape=( self.topology[layer] ), dtype=self.float_precision )
            
            # generate L bias-layers
            self.biases[layer] = np.array([0. for i in range(self.topology[layer])], dtype=self.float_precision)
        

        # compute size
        self.parameters = 0
        for i in range(1, self.L+1):
            self.parameters += ( self.topology[i] * self.topology[i-1] + self.topology[i] )

    def initialize (self, method: str='random', weight_range: list=[-.1, .1], bias_range: list=[-.1, .1], 
                    mean_weight:float=0, var_weight:float=0, mean_bias:float=0, var_bias:float=0): # works

        '''
        Initializes the networks weight tensors and bias vectors with provided method.

        Methods:
        random, uniform -> use range args
        norm, normal    -> use var and mean args
        '''
        
        for layer in range(1, self.L + 1):

            for i in range(self.topology[layer]):

                for j in range(self.topology[layer-1]):
                
                    if method in ['random', 'uniform']:
                        self.tensors[layer][i][j] = np.random.uniform(weight_range[0], weight_range[1])
                    
                    elif 'norm' in method:
                        self.tensors[layer][i][j] = np.random.normal(mean_weight, var_weight)

                # fill biases in layer vector
                if method in ['random', 'uniform']:
                    self.biases[layer][i] = np.random.uniform(bias_range[0], bias_range[1])

                elif 'norm' in method:
                    self.biases[layer][i] = np.random.normal(mean_bias, var_bias)

        self.initialized = True

    def load (self, filepath: str|Path) -> None:

        '''
        filepath    Absolue path to file or target director , in the ladder case a filename 
                    will be generated from model name.
        compress    Saves model in compressed state.
        override    If an old dump with provided filepath exists, it will be overwritten.
        '''
        
        try:

            # convert filepath to path object
            if type(filepath) is str:
                filepath = Path(filepath)
            
            if filepath.suffix != '.npz':
                filepath = filepath.with_suffix('.npz')
            
            # short-cut if path is not existing
            if not filepath.exists():
                print(f'[error]: the provided filepath "{filepath.resolve()}" does not exists!')
                return
        
            # try to load
            print(f'[load]: load model from file {filepath.resolve()} ...') 
            loaded_data = np.load(filepath, allow_pickle=True)

            restored_dict = {key: loaded_data[key].tolist() if type(loaded_data[key]) is np.ndarray else loaded_data[key] for key in loaded_data.keys()}

            # re-construct model
            self.sequential(
                restored_dict['name'],
                list(restored_dict['topology']),
                list(restored_dict['activations'])
            )

            # apply weights and biases
            self.tensors = dict(restored_dict['tensors'])
            self.biases = dict(restored_dict['biases'])

            self.L = restored_dict['layer']

            # restore infos about fit
            self.fit_label = restored_dict['fit']

            # make a test prop
            self.propagate(np.zeros((self.topology[0],)))

            print(f'[load]: successfully loaded model "{self.name}".')

        except Exception as e: 

            print('[load]: could not load the dump:', e)
            print_exc()
    
    def save (self, filepath: str|Path, compress:bool=True, override:bool=False) -> None:

        '''
        filepath    Absolue path to file or target directory, in the ladder case a filename 
                    will be generated from model name.
        compress    Saves model in compressed state.
        override    If an old dump with provided filepath exists, it will be overwritten.
        '''

        try:

            # convert filepath to path object
            if type(filepath) is str:
                filepath = Path(filepath)

            # check if provided filepath already exists and check for override
            if filepath.is_file() and filepath.exists():
                print(f'[save]: filepath "{filepath.resolve()}" already exists ...')
                if override:
                    print(f'[save]: will override "{filepath.resolve()}" ...')
                else:
                    print(f'[save]: If the file should be overriden, set the override flag to true: CNN.save(..., override=True, ...)')
                    print(f'[save]: Failed to save the model.')
                    return
                
            # check if path is a directory, if so generate a filename
            if filepath.is_dir():
                filename = filename = self.name + '.npz'
                filepath = filepath.joinpath(filename)
                print(f'[save]: provided filepath is a directory, will use model name as file name: "{filename}" ...')

            # check for correct file suffix
            elif filepath.suffix != '.npz':
                filepath = filepath.with_suffix('.npz')

            # package the output
            dictionary = {
                'name' : self.name,
                'tensors' : self.tensors,
                'biases' : self.biases,
                'activations': self.activation_overlay,
                'topology' : self.topology,
                'layer' : self.L,
                'fit': self.fit_label
            }
            
            # save the dictionary
            compress_string = ''
            if compress:
                np.savez_compressed(filepath, **dictionary)
                compress_string = ' compressed'
            else:
                np.savez(filepath, **dictionary)

            print(f'[save]: successfully saved{compress_string} model "{self.name}" to {filepath.resolve()}')

        except Exception as e:

            print('[save]: an error occured:', e)
    
    def transpose (self, tensor: np.ndarray, layer: np.ndarray, bias: np.ndarray|None=None, activation: str='linear'): # works

        '''
        A single layer transposition given by
        activation( tensor * layer + bias )

        Batch gradient descent will demand propagating a whole tensor
        i.e. layer becomes 2-dimensional tensor, then the output will be a tensor
        with shape (batch_size, input_size).
        '''

        if len(layer.shape) == 1:
            return self.activate(tensor.dot(layer) + bias, model=activation)
        else:
            # We need to transpose the input tensor to align shapes in the dot product
            # and then transpose the result back to add bias to all elements in first axis
            # which is the layer activation vector.
            return self.activate((tensor.dot(layer.T)).T + bias, model=activation)
    
    def propagate_to_layer (self, _layer: int, _input: np.ndarray) -> np.ndarray: # works

        '''
        Propagates provided input up until the desired layer.
        Respects corresponding activations and biases along layers.
        Returns the activation array of the _layer-th layer.
        '''

        c = self.activate(_input, model=self.activation_overlay[0])

        for layer in range(1, _layer + 1):

            c = self.transpose(self.tensors[layer], c, bias=self.biases[layer], activation=self.activation_overlay[layer])

        return c
    
    def propagate (self, _input: np.ndarray) -> np.ndarray: # works

        '''
        Uses CNN.propagate_to_layer to propagate an input from input to output layer.
        The return is the same as for CNN.propagate_to_layer.
        '''

        # ensure the input is vectorized
        if type(_input) == list:

            _input = np.array(_input)

        return self.propagate_to_layer(self.L, _input)

    def forward (self, _input: np.ndarray) -> np.ndarray:

        '''
        Alias for propagate method.
        '''

        return self.propagate(_input)
    
    def activate (self, x: np.ndarray|float, derivative: bool=False, model: str='sigmoid') -> np.ndarray|float: # works

        # Sigmoid or logistics function
        # https://en.wikipedia.org/wiki/Sigmoid_function
        if model == 'sigmoid':

            sig = lambda x: 1 / ( 1 + np.exp(-x) )

            if derivative:

                return sig(x) * (1 - sig(x))

            return sig(x)
        
        # ReLU activation function
        # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        elif model == 'relu':

            if type(x) == np.ndarray or type(x) == list:

                out = np.maximum(0, x)  # Apply ReLU element-wise

                if derivative:
                    out[x <= 0] = 0  # Derivative of ReLU
                    out[x > 0] = 1
                    
                return out

        # Identical function, will simply return the input
        elif model == 'linear':

            if derivative:
                
                if type(x) == np.ndarray or type(x) == list:

                    return np.ones(x.shape)
                
                return 1.
            
            return x
        
        # Softmax activation normalized and boltzmann-weighted to fulfill probability condition.
        # https://en.wikipedia.org/wiki/Softmax_function
        elif model == 'softmax':

            boltzman_factor = np.exp( x )
            
            if len(x.shape) > 1:
                # activate linear prediction with softmax
                
                dist = (boltzman_factor.T / np.sum(boltzman_factor, axis=1)).T
            
            else:

                dist = (boltzman_factor / np.sum(boltzman_factor)) 


            if derivative:

                # Compute the outer product of the softmax vector with itself
                outer = np.einsum('ij,ik->ijk', dist, dist)

                # Create a 3D diagonal matrix with the softmax values on the diagonals
                diag = np.einsum('ij,jk->ijk', dist, np.eye(dist.shape[1]))

                # Subtract the outer product from the diagonal matrix
                jacobian_m = diag - outer

                # Compute the determinant of each 2D matrix in the 3D stack
                return jacobian_m

            return dist

    def loss (self, output: np.ndarray, target: np.ndarray, evaluation_criterion: Callable=MeanSquaredError()): # works
        
        '''
        Loss function based on output and target vector.
        Evaluation happens according to provided evaluation function.
        Default: error.mean_squared_error
        '''
        
        return evaluation_criterion.evaluate(output, target)

    def nudge (self, layer: np.ndarray, i: int=0, j: int=0, change:int|np.ndarray=0, object: str='weights') -> None:

        '''
        Allows to nudge weight tensors and bias vectors (corresponding to layer) either by
        applying a scalar change to weight (using i,j indices) or by applying a whole delta 
        tensor - then change=np.ndarray. The 'object' type determines wether the change 
        should be applied to either bias or weight.
        '''

        if object == 'weights':

            if type(change) in [np.ndarray, list]:
                self.tensors[layer] = self.tensors[layer] + change
            else:
                self.tensors[layer][i][j] = self.tensors[layer][i][j] + change
        
        if object == 'biases':
            
            if type(change) in [np.ndarray, list]:
                self.biases[layer] = self.biases[layer] + change
            else:
                self.biases[layer][i] = self.biases[layer][i] + change

    def fit (self, samples: list[list[np.ndarray, np.ndarray]], batch_size: int=1, epochs: int=1, verbose: bool=False, 
            error_plot: bool=False, stop: float=0.0, learning_decay: float=0.0, learning_rate: float=None, shuffle: bool=True, 
            evaluation_criterion: Criterion=MeanSquaredError(), split: float|None=None, test_samples: list[list[np.ndarray, np.ndarray]]|None=None, 
            epoch_callback: Callable|None=None) -> dict[str, any]:
            
            '''
            Main fully-customizable fitting method.

            batchSize               if 1 then its stochastic gradient decent, if its equal len(sample) its batch g. d., everything else is mini-batch
            eochs                   Number of training epochs.
            verbose                 verbose training output.
            error_plot              Whether to plot the training loss at the end.
            stop                    Loss function value at which to stop training.
            learning_decay          The amount the learning rate will decay after each epoch.
            learning_rate           if 0.0 then the predefined learning rate will be taken else it will be overwritten.
            shuffle                 If to shuffle the training set between epochs.
            evaluation_criterion    Evaluation function for evaluating the loss
            split                   fractional split [.0, 1.] of training set e.g. 0.8 means that 80% of 
                                    training samples will be used, rest 20% will be used for testing.
            test_samples            A list of (input, output) tuples -same as samples.
                                    To pass a persistent training set, without the need of split method.                        
            epoch_callback          A callback function func(client) which will be called after every epoch.
                                    The function takes the client as argument.
            '''

            _skip_errors = False
            _diverged = False
            _stop_triggered = False

            samples_size = len(samples) if type(samples) is list else samples.shape[0]

            # split provided samples into training and test set
            if split:
                if shuffle:
                    # shuffle before splittling the set
                    np.random.shuffle(samples) 
                # determine the split index
                training_size = int( samples_size * split )
                test_size = samples_size - training_size
                print(f'[fitting]: split the {samples_size} training samples to {training_size} for training and {test_size} for testing ...') if verbose else None
                # create final training and test sample
                test_samples = samples[training_size:]
                training_samples = samples[:training_size]
            else:
                training_size = samples_size
                test_size = 0
                training_samples = samples
            
            # compute number of batches formable
            if batch_size == 1:
                # stochastic gradient descent
                batches_number = training_size
            else:
                # mini batch gradient descent
                batches_number = int( training_size / batch_size )

            # override global hyper parameters
            if learning_rate: 
                self.learning_rate = learning_rate
            self.learning_decay = learning_decay
            
            # create cache for loss accumulation
            losses = []
            # training epochs
            for epoch in range(1, epochs+1):

                print(f'[fitting]: ==== Epoch  {epoch}/{epochs} ====') if verbose else None

                epoch_loss = 0
                
                # re-arrange shuffled training samples into x and y sets
                x_train, y_train = [], []
                for s in training_samples:
                    x_train.append(s[0])
                    y_train.append(s[1])
                x_train, y_train = np.array(x_train), np.array(y_train)

                # back-propagate batch-wise
                for b in range(batches_number):

                    try:

                        # slice out the b-th batch out of training_samples
                        x_batch, y_batch = x_train[ b*batch_size : (b+1)*batch_size ], y_train[ b*batch_size : (b+1)*batch_size ]

                        # backpropagate batch
                        # backprop_result = self.backprop2(x_batch, y_batch, evaluation_function, evaluation_kwargs)
                        backprop_result = self.backprop(x_batch, y_batch, evaluation_criterion=evaluation_criterion)

                        # check if loss is divergent
                        if f'{backprop_result["loss"]}'.lower() == 'nan' or np.isnan(backprop_result["loss"]) or isnan(backprop_result["loss"]):
                            print('[fitting]: the fit diverged to infinity which lead to an undefined cost value. To resolve this issue it could be helpful to take smaller learning rates.\nStop fitting.')
                            _diverged = True
                            break

                        # denote result
                        # bias_deltas.append(backprop_result['biases'])
                        # weight_deltas.append(backprop_result['weights'])
                        # losses.append(backprop_result['loss'])
                        epoch_loss += backprop_result["loss"]

                        # verbpose status output to console
                        print(f'[fitting]: epoch {epoch}/{epochs}; batch {b+1}/{int(training_size/batch_size)}; cost={backprop_result["loss"]}') if verbose else None
                        
                    except Exception as e:

                        if not _skip_errors:

                            raise e
                        
                        print(f'[fitting]: epoch {epoch}/{epochs}; batch {b}/{int(training_size/batch_size)} --> Error') if verbose else None
                
                # denote the avg. batch loss across the whole epoch
                avg_batch_loss = epoch_loss / batches_number
                losses.append(avg_batch_loss)

                # check if loss has reached a threshold
                if stop and losses[-1] < stop:
                # if stop and backprop_result['loss'] < stop and np.mean(losses[-batch_size:]) <= stop:
                    print(f'[fitting]: stop fitting as loss reached stopping threshold ({losses[-1]} <= {stop})')
                    _stop_triggered = True

                # catch if a terminating event triggered
                # to stop the training here before applying any faulty changes
                if _stop_triggered or _diverged:
                    break

                # apply decay to learning rate 
                # important after every epoch
                if self.learning_decay:
                    self.learning_rate *= (1-self.learning_decay)

                # finally re-shuffle the training set
                if shuffle:
                    np.random.shuffle(training_samples)
                
                # run an epochal callback function
                if epoch_callback:
                    epoch_callback(self)
                
                # collect garbage
                gc.collect()

            # testing 
            if test_samples:
                accuracy = self.test(test_samples, mode='argmax')
                self.accuracy = accuracy
                # successes = 0
                # for s in test_samples:
                #     # make a maximum likelihood prediction
                #     if np.argmax(s[1]) == np.argmax(self.propagate(s[0])):
                #         successes += 1
                # accuracy = np.round(100 * successes / test_size, 2)
                print(f'[testing]: categorization accuracy: {accuracy}%')
            
            # plot loss function for analysis
            if error_plot:
                plt.plot(losses)
                plt.grid()
                plt.show()

            # collect statistics
            fit_information = {
                'accuracy_testing': accuracy if split else '-',
                'batch_size': batch_size,
                'epochs': epochs, 
                'learning_decay': self.learning_decay,
                'learning_rate': self.learning_rate,
                'loss': losses,
                'samples_total': samples_size,
                'split': split
            }

            # denote fit information in a label
            self.fit_label = fit_information
            
                
            return fit_information

    def backprop (self, x_batch: np.ndarray, y_batch: np.ndarray, evaluation_criterion: Callable=MeanSquaredError()) -> dict:

        return self.backprop5(x_batch, y_batch, evaluation_criterion)

    def backprop5 (self, x_batch: np.ndarray, y_batch: np.ndarray, evaluation_criterion: Callable=MeanSquaredError()) -> dict:

        '''
        Back-propagation algorithm with gradient descent,
        fully parallelized by vectorized calculus.

        [Return]
        Dictionary with weights, biases, and loss array.
        '''

        # parameters
        batch_size = x_batch.shape[0]

        # propagate the whole batch up to every layer
        # outputs = (layer[list], (batch, output)[array])
        outputs = [x_batch]
        for layer in range(1, self.L+1):
            outputs.append( self.transpose(self.tensors[layer], outputs[-1], self.biases[layer], activation=self.activation_overlay[layer]) )

        # track consecutive differences
        last_layer_delta = None

        # iterate from L to 1
        for layer in range(self.L, 0, -1):

            # corresponding tensor to current layer
            tensor_current = self.tensors[layer]

            # span the gradient elements once
            weight_gradient = np.ndarray( shape=( tensor_current.shape[0], tensor_current[0].shape[0] ) )
            bias_gradient = np.ndarray( shape=(tensor_current.shape[0]) )

            # get the values of prior layer
            layers_prior = outputs[layer-1]

            # compute the difference in consecutive layer values, 
            # i.e. l and l+1, if l=L, then l+1=target
            if layer == self.L:

                # layers_current = self.transpose(tensor_current, layers_prior, bias=self.biases[layer], activation='linear')
                # We exploit the fact that the result of layer L is known already accessible within the provided y batch.
                # Time is saved as the layer output does not have to be determined.
                layers_current = outputs[layer] # current layer output is the final output
                # compute the delta of output layer and target -> shape=(batchsize, outputsize)
                # using the corresponding evaluation function.
                # for MSE: delta = 2 * (layers_current - y_batch) 
                delta = evaluation_criterion.evaluate(layers_current, y_batch, derivative=True)

            else:

                layers_current = self.transpose(tensor_current, layers_prior, bias=self.biases[layer], activation=self.activation_overlay[layer-1])
                tensor_post = self.tensors[layer+1]
                layers_post_raw = self.transpose(tensor_post, layers_current, self.biases[layer+1], activation='linear')
                layers_post_derivative = self.activate(layers_post_raw, derivative=True, model=self.activation_overlay[layer])
                
                # apply post tensor to post layers activation values 
                # -> the resulting array holds a batch of input arrays which each need to be multiplied element-wise with prior differences
                delta = ( tensor_post.T @ ( layers_post_derivative * last_layer_delta ).T ).T

            # remember only the last delta
            last_layer_delta = delta

            # compute derivative of activation function of the weighted sum of layer l
            sigma_prime_batch = self.activate(layers_current, True, model=self.activation_overlay[layer-1])

            # From here on we treat each batch dimension separately.
            # Iterate for every input in the batch
            batch_weight_gradients = np.zeros( shape=( tensor_current.shape[0], tensor_current[0].shape[0] ) )
            batch_bias_gradients = np.zeros( shape=(tensor_current.shape[0]) )

            # iterate batches with gradient descent
            # The vectorization of the weight change matrix will be advantageous for performance.
            for sample in range(batch_size):

                prior = layers_prior[sample]
                   
                
                # compute derivative of activation function of the weighted sum of layer l
                sigma_prime = sigma_prime_batch[sample]

                # compute gradient tensor in vectorized way, which is basically:
                # weight_change[i][j] = (-1) * self.learning_rate * ( (1 - self.momentum) * ( prior[j] * sigma_prime[sample][i] * delta[sample][i] ) + self.momentum * self.last_weight_change[layer][i][j] ) 
                weight_gradient = (1 - self.momentum) * ( np.outer(prior, sigma_prime) * delta[sample] ).T + self.momentum * self.last_weight_change[layer]
                
                # remember the weight change for next iteration
                self.last_weight_change[layer] = weight_gradient

                # bias vector is obtained from element-wise multiplication of sigma_prime and delta vector
                bias_gradient = sigma_prime * delta[sample]
                
                # accumulate batch change
                batch_weight_gradients += weight_gradient
                batch_bias_gradients += bias_gradient

            # apply the batch-normalized gradients to layer tensor and bias vector
            # also apply the learning rate at the end
            self.weight_gradients[layer] = - ( self.learning_rate / batch_size ) * batch_weight_gradients 
            self.bias_gradients[layer] = - ( self.learning_rate / batch_size ) * batch_bias_gradients
        
        # propagation error/loss
        loss = self.loss(outputs[-1], y_batch, evaluation_criterion=evaluation_criterion)

        # finally apply gradient tensors to weights and biases
        for layer in range(1, self.L+1):
            self.nudge(layer, change=self.weight_gradients[layer])
            self.nudge(layer, change=self.bias_gradients[layer], object='biases')

        # --- Return demanded --- #
        return {'weights': self.weight_gradients, 'biases': self.bias_gradients, 'loss': loss}
    
    def size (self):

        '''
        Alias of global size method, applied to inner parameters.

        Returns the size of provided tensor in 

        bytes       if individual_element_size=False
        bits        if individual_element_size=True as 
                    it will return the single element size
                    which equals the dtype
        '''

        s = 0
        for layer in range(1, self.L + 1):
            s += size(self.tensors[layer]) + size(self.biases[layer])
        
        return s

    def test (self, samples: list[list[np.ndarray, np.ndarray]], mode: str='argmax', threshold: float=0) -> float:

        '''
        Returns the prediction accuracy.

        mode        argmax - for categorization e.g. one-hot encoded distribution outputs
                    distance - magnitude distance (also provide threshold argument)
                    mse - mean squared distance (also provide threshold argument)
                    loss - avg. loss per sample (also provide loss function)

        '''

        successes = 0

        for s in samples:
            # make a maximum likelihood prediction
            if mode == 'argmax':
                if np.argmax(s[1]) == np.argmax(self.propagate(s[0])):
                    successes += 1
            elif mode == 'distance':
                if np.abs(s[1] - s[0]).sum() < threshold:
                    successes += 1
            elif mode == 'mse':
                if ((s[1] - s[0])**2).mean() < threshold:
                    successes += 1
            
        accuracy = np.round(100 * successes / len(samples), 2)

        return accuracy

# Long Short-term Memory Neural Module
# =========================================
class LSTM (Module):

    '''
    Unrollable LSTM module converged to a layer-like structure.
    General topology, assuming x is the input and y the output:

                hidden      state
                   |          | 
        x_0 ---> |     LSTM     | ---> y_0
                   |          | 
        x_1 ---> |     LSTM     | ---> y_1
                   |          | 
        x_2 ---> |     LSTM     | ---> y_2
                         .
                         .
                         .
    
    x and y form the input (vector received from former layer) 
    and output (vector passed on to further layers), respectively.
    '''

    def __init__ (self, state_init_method: str='cold', weight_init_method: str='cold') -> None:

        super().__init__()


        '''
        c_{t-1} -------- x --------------- + -----------------┬----> c_{t}
                         |                 |                  |
                         |        ┌------- x                 tanh
                         |        |        |                  |
                     sigmoid  sigmoid    tanh    sigmoid ---> x
                         |        |        |        |         |
                        wi_1     wi_2    wi_3      wi_4       |
        h_{t-1} ---- + --┴--------┴--------┴--------┘         └----> h_{t} (y_output)
                     |
                  x_input
        '''

        # current cell and hidden state of the unit
        if state_init_method in ['random', 'hot']:
            self.cell = np.random.uniform(-.5, .5)      # long-term state
            self.hidden = np.random.uniform(-.5, .5)    # short-term memory state
        else:
            self.cell = 0         # long-term state
            self.hidden = 0       # short-term memory state

        # initialize weights and biases
        if weight_init_method in ['random', 'hot']:
            # first row: input weights which go into the 4 stages (labeled wi)
            # second row: short term or "recursive" weights 
            self.weights = np.random.uniform(-.5, .5, size=(2,4))
            self.bias = np.random.uniform(-.5, .5, size=(4,))
        else:
            self.weights = np.zeros(shape=(2,4))
            self.bias = np.zeros(shape=(4,))
        
    def __call__ (self, _input: np.ndarray) -> np.ndarray:

        '''
        The LSTM module is callable with unroll method.
        
        [Parameters]
        _input      ndarray

        [Return]
        Returns ndarray of same shape as _input.
        '''

        return self.unroll(_input)

    def unroll (self, _input: np.ndarray) -> np.ndarray:

        '''
        Forward method for arbitrary sequence lengths.
        Accepts an ndarray "timeseries" of arbitrary shape into unrolled feedback loop, 
        where the 0-th shape dimension will be interpreted as the input/timeseries index.
        All further dimenions/indices are attributed to the internal structure
        of the input data. -> (timeseries_index, *object_dimensions)

        [Parameters]
        _input      ndarray

        [Return]
        Returns ndarray of same shape as _input.
        '''

        # the input elements of the timeseries are aranged along the first dimension
        output = np.zeros(shape=_input.shape)

        # Unrolled feedback propagation across consecutive stages in time.
        # I.e. every output neuron in the output layer will depend on the one before.
        for i in range(_input.shape[0]):
            output[i], self.cell, self.hidden = self.forward(_input[i], self.cell, self.hidden)
        
        return output

    def forward (self, _input: np.ndarray, cell: float, hidden: float) -> tuple[np.ndarray, float, float]:

        '''
        Forwards a single element x_0 of e.g. a timeseries x.
        Calling this method has no impact on internal state parameters.

        [Parameters]
        _input      Single input, e.g. the first element of a timeseries.
        cell        Input cell value i.e. the most recent cell value.
        hidden      Recent hidden state value.

        [Return]
        tuple (_ouput, new cell value, new hidden value)
        '''

        # intermediate sigmoid values
        xw = _input * self.weights[0]
        hr = hidden * self.weights[1]

        # junction of weighted inputs, weighted short mem, and bias
        sum_xhw_hr_b = xw + hr + self.bias

        # activate in two steps
        # 1. activate all with sigmoid for performance
        activated_sums = activate(sum_xhw_hr_b) 
        # 2. override 3. stage (2nd index) with tanh activation
        activated_sums[2] = activate(sum_xhw_hr_b[2], model='tanh')

        # compute new states
        c = cell * activated_sums[0] + activated_sums[1] * activated_sums[2]
        h = activate(c, model='tanh') * activated_sums[3]
        # the hidden value is also the output value of the block

        return h, c, h


# Toolkit
# =========================================
class Loader:

    '''
    Loader module for loading models and data.
    '''

    def from_pretrained (filepath: str|Path) -> Network:

        '''
        Loads a pre-trained model from a saved .npz file.

        [Return]
        A loaded network.
        '''

        rohling = CNN()
        rohling.load(filepath)

        return rohling

    def load_image_data (dirpath: str|Path, suffix: str='', class_string_separator: str='_', separator_index: int=1, 
                         one_hot_encoded: bool=True, normalize: bool=True, color_band: int|None=None, rectify_mixed_background: bool=False, invert: bool=False) -> list[list[np.ndarray, np.ndarray]]:

        '''
        Loads images into pyron-compliant format 
        i.e. -> [[x_array_input_1, x_array_output_1], [x_array_input_2, x_array_output_2], ...]
        by flattening the pixel values onto a 1D input vector. 

        The output vector, which is one-hot encoded, will be drawn from the file name. For this the
        digit is parsed from the filename by seprators (default '_')
        e.g. c_0_id16.png -> class 0

        Provide suffix e.g. '.png' for correct file type filtering.

        [Return]
        Sample dataset list[list[ndarray, ndarray]].
        '''

        if type(dirpath) is str:
            dirpath = Path(dirpath)

        # load dataset
        dataset = []
        files = [entry for entry in dirpath.iterdir()]

        for file in files:

            if suffix and file.suffix != suffix:
                continue

            # filePath = os.path.join(dirpath, file)
            filePath = dirpath.joinpath(file)
            im_frame = Image.open(filePath)
            x_vector = np.array(im_frame.getdata(color_band))

            # pixel array, each value is in [0,100]
            # normalize 0-100 to 0-1 (Min Max Scaling).
            if normalize:
                x_vector = x_vector / np.max(x_vector)

            # invert colors
            if rectify_mixed_background and x_vector.mean() > 0.5:
                x_vector = 1 - x_vector
            if invert:
                x_vector = 1 - x_vector

            # extract label or "y" value from filename
            if one_hot_encoded:
                label = int(filePath.name.split(class_string_separator)[separator_index])
                y_vector = np.zeros((10,))
                y_vector[label] = 1.
            else:
                # otherwise return 0-dim. integer labels
                y_vector = np.array([label])

            dataset.append([x_vector, y_vector])
        
        return dataset


# Global Methods 
# =========================================
def activate (x: np.ndarray|float, derivative: bool=False, model: str='sigmoid') -> np.ndarray|float: # works

    '''
    Standalone activate function.
    model       linear, sigmoid, relu, softmax
    '''

    # Sigmoid or logistics function
    # https://en.wikipedia.org/wiki/Sigmoid_function
    if model == 'sigmoid':

        sig = lambda x: 1 / ( 1 + np.exp(-x) )

        if derivative:

            return sig(x) * (1 - sig(x))

        return sig(x)

    # Hyperbolic Tangent "tanh"
    # https://en.wikipedia.org/wiki/Hyperbolic_functions
    if model == 'tanh':
        
        tanh_x = np.tanh(x)

        if derivative:

            return 1 - tanh_x * tanh_x

        return tanh_x
    
    # ReLU activation function
    # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    elif model == 'relu':

        if type(x) == np.ndarray or type(x) == list:

            out = np.maximum(0, x)  # Apply ReLU element-wise

            if derivative:
                out[x <= 0] = 0  # Derivative of ReLU
                out[x > 0] = 1
                
            return out

    # Identical function, will simply return the input
    elif model == 'linear':

        if derivative:
            
            if type(x) == np.ndarray or type(x) == list:

                return np.ones(x.shape)
            
            return 1.
        
        return x
    
    # Softmax activation normalized and boltzmann-weighted to fulfill probability condition.
    # https://en.wikipedia.org/wiki/Softmax_function
    elif model == 'softmax':

        boltzman_factor = np.exp( x )
        
        if len(x.shape) > 1:
            # activate linear prediction with softmax
            
            dist = (boltzman_factor.T / np.sum(boltzman_factor, axis=1)).T
        
        else:

            dist = (boltzman_factor / np.sum(boltzman_factor)) 


        if derivative:

            # Compute the outer product of the softmax vector with itself
            outer = np.einsum('ij,ik->ijk', dist, dist)

            # Create a 3D diagonal matrix with the softmax values on the diagonals
            diag = np.einsum('ij,jk->ijk', dist, np.eye(dist.shape[1]))

            # Subtract the outer product from the diagonal matrix
            jacobian_m = diag - outer

            # Compute the determinant of each 2D matrix in the 3D stack
            return jacobian_m

        return dist

def clone_training (model: Network, samples: list[list[np.ndarray]], clone_number: int=1, batch_size: int=1, epochs: int=1, stop: float=0.0, 
            learning_decay: float=0.0, learning_rate: float=None, shuffle: bool=True, evaluation_criterion: Callable=MeanSquaredError(), 
            split: float|None=None, epoch_callback: Callable|None=None, epochs_warmup: int|None=None, loss_weighting: bool=False,
            init_method:str='random', weight_range: list=[-.1, .1], bias_range: list=[-.1, .1], mean_weight:float=0, var_weight:float=0, mean_bias:float=0, var_bias:float=0):

    '''
    Clone training - a random forrest approach.
    '''

    # 0. Instantiate a model for every clone of the g clones.
    g = clone_number
    clones, threads = [copy.deepcopy(model) for _ in range(g)], []

    # 1. Shuffle the set to ensure homogeneous and thus uniform distrbution in all subsets.
    np.random.shuffle(samples)

    # 2. Split samples to g equally sized samples.
    sample_size = len( samples )
    split_size = int( sample_size / g )
    clone_samples = [samples[i*split_size:(i+1)*split_size] for i in range(g)]

    # test
    # for clone_sample in clone_samples:
    #     print(len(clone_sample))

    
    # 3. Alpha Correlation
    # warmup to correlate the clones
    if epochs_warmup:
        print(f'[clone training]: warmup for {epochs_warmup} epochs -> alpha selection ...')
        for i in range(g):

            # initialize each clone independently
            clones[i].initialize(init_method, weight_range, bias_range, mean_weight, var_weight, mean_bias, var_bias)

            # wrap the thread in a lambda call
            training_thread = lambda : clones[i].fit(
                clone_samples[i],
                batch_size=batch_size,
                epochs=epochs_warmup,
                verbose=False,
                error_plot=False,
                stop=stop,
                learning_decay=learning_decay,
                learning_rate=learning_rate,
                shuffle=shuffle,
                evaluation_criterion=evaluation_criterion,
                split=split,
                epoch_callback=epoch_callback
            )

            # train each clone in a separate thread with one of the splitted samples
            # (this is a simulation to g separate GPUs)
            prc = thread(function=training_thread)

            # store and start
            threads.append(prc)
            prc.start()
        
        # await all threads
        for t in threads:
            while not t.finished:
                sleep(.1)
        
        # pick the one with greatest test accuracy
        alpha = None
        for clone in clones:
            if not alpha:
                alpha = clone
                continue
            if clone.accuracy > alpha.accuracy:
                alpha = clone
        
        # all clones become a new clone of the alpha
        print(f'[clone training]: alpha successfully selected, correlate clones ...')
        clones = [copy.deepcopy(alpha) for _ in range(g)]

    # 4. Pass a split set to each clone for parallelized training
    print(f'[clone training]: train {g} independent clones over {epochs} epochs, this may take a while ...')
    for i in range(g):

        # initialize each clone independently
        clones[i].initialize(init_method, weight_range, bias_range, mean_weight, var_weight, mean_bias, var_bias)

        # wrap the thread in a lambda call
        training_thread = lambda : clones[i].fit(
            clone_samples[i],
            batch_size=batch_size,
            epochs=epochs,
            verbose=False,
            error_plot=False,
            stop=stop,
            learning_decay=learning_decay,
            learning_rate=learning_rate,
            shuffle=shuffle,
            evaluation_criterion=evaluation_criterion,
            split=split,
            epoch_callback=epoch_callback
        )

        # train each clone in a separate thread with one of the splitted samples
        # (this is a simulation to g separate GPUs)
        prc = thread(function=training_thread)

        # store and start
        threads.append(prc)
        prc.start()
    
    # await all threads
    for t in threads:
        while not t.finished:
            sleep(.1)

    # 5. Extract weighting from the corresponding losses
    if split and loss_weighting:
        loss_dist = np.array([clone.accuracy for clone in clones])
        loss_weights = activate(loss_dist, model='softmax')
    else:
        loss_weights = np.ones((len(clones),))

    # 6. Merge all weights by sample mean into a new model corpus.
    corpus = CNN()
    corpus.sequential(
        model.name,
        model.topology,
        model.activation_overlay
    )

    # take the loss-weighted mean of all weights and biases
    # i.e. clones which performed better will be taken more into account
    for clone in clones:
        for layer in range(1, model.L+1):
            print('clone weight snip out - ', clone.tensors[layer][0][:3]) if layer == 3 else None
            corpus.tensors[layer] += loss_weights[layer-1] * clone.tensors[layer] / g
            corpus.biases[layer] += loss_weights[layer-1] * clone.biases[layer] / g
    print('merged corpus snip out - ', corpus.tensors[3][0][:3]) 


    return corpus

def sample (shape: tuple, max: float=1) -> np.ndarray:

    '''
    Sample a random ndarray.
    '''

    return np.random.uniform(-max, max, size=shape)

def size (tensor: np.ndarray, individual_element_size: bool=False, verbose: bool=False) -> int:

    '''
    Returns the size of provided tensor in 

        bytes       if individual_element_size=False
        bits        if individual_element_size=True as 
                    it will return the single element size
                    which equals the dtype
    '''

    precision = tensor.dtype
    
    empty_shape = []
    for i in range(len(tensor.shape)):
        empty_shape.append(tensor.shape[i])
    empty_shape[-1] = 0
    empty = np.empty(empty_shape) # empty 

    # count the parameters by multiplying dimensions
    parameters = 1
    for d in range(len(tensor.shape)):
        parameters *= tensor.shape[d]

    size_empty = sys.getsizeof(empty)
    size = sys.getsizeof(tensor)
    stored_size = size - size_empty

    if verbose:
        print(f'size {precision} empty', size_empty, 'bytes')
        print(f'size {precision}', size, 'bytes')
        print(f'size storage {precision}', stored_size, 'bytes')
        print(f'storage size per element', int(stored_size / parameters * 8), 'bit')
    
    if individual_element_size:
        return stored_size
    return size