import numpy as np
import random, time
import matplotlib.pyplot as plt
import json

class error:

    def chi_squared(self, xarray, yarray):
        delta = xarray - yarray
        return delta.dot(delta) / xarray.shape[0]


class network:

    def __init__(self):

        # object varibles
        self.learningRate = 0.01
        self.learning_decay = 0.00
        self.momentum = 0.05
        self.L = 0
        self.tensors = {}
        self.changes = {}
        self.biases = {}
        self.topology = []
        self.layerSpace = []
        self.activation = 'sigmoid'
        self.error = 0
        self.name = ''
        
    def new(self, name='new_network', topology=list(), activation_overlay=None): # works

        '''
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
        if activation_overlay == None:

            self.activation_overlay = ['sigmoid' for i in range(self.L+1)]
            self.activation_overlay[0] = 'id'
            self.activation_overlay[-1] = 'id'
        
        else:

            if len(activation_overlay)-1 != self.L:

                raise ValueError('the provided activation overlay must have same dimension as the total layer number.')

            self.activation_overlay = activation_overlay

        for layer in range(1, self.L + 1):

            # generate only L-1 transfer matrices
            self.layerSpace.append(layer)
            self.tensors[layer] = np.zeros( shape=( self.topology[layer], self.topology[layer-1] ) )
            self.changes[layer] = np.zeros( shape=( self.topology[layer], self.topology[layer-1] ) )
            
            # generate L bias-layers
            self.biases[layer] = np.array([0. for i in range(self.topology[layer])])

    def initialize(self, weight_range=[-.1, .1], bias_range=[-.1, .1]): # works
        
        for layer in range(1, self.L + 1):

            for i in range(self.topology[layer]):

                for j in range(self.topology[layer-1]):
                
                    # fill weight in (l, i, j) weight tensor
                    self.tensors[layer][i][j] = np.random.uniform(weight_range[0], weight_range[1])

                # fill biases in layer vector
                self.biases[layer][i] = np.random.uniform(bias_range[0], bias_range[1])

    def load(self, filename=None):
        
        try:
            if filename == None:
                filename = 'new_network.json'
            with open(filename, 'r') as infile:
                dictionary = eval(infile.read().replace('array', 'np.array'))
                self.name = dictionary['name']
                self.tensors = dictionary['tensors']
                self.biases = dictionary['biases']
                self.activation_overlay = dictionary['activations']
                self.topology = dictionary['topology']
                self.L = dictionary['layer']
                print('[load]: {} loaded.'.format(filename))
        except Exception as e: 
            print('[load]: could not load the dump. ', e)
    
    def dump(self, filename=None):

        '''
        If an old dump exists, it will be overwritten.
        '''

        try:

            # package the output
            dictionary = {
                'name' : self.name,
                'tensors' : self.tensors,
                'biases' : self.biases,
                'activations': self.activation_overlay,
                'topology' : self.topology,
                'layer' : self.L
            }

            if filename == None:
                filename = self.name + '.nf'
            with open(filename, 'w+') as f:
                f.write("""{}""".format(dictionary))
                print('[dump]: dumped to {}.'.format(filename))

        except Exception as e:

            print('[dump]: during dumping an error occured:', e)
    
    def transpose(self, tensor, layer, bias=None, activation='id'): # works

        if type(bias) == np.ndarray:

            return self.activate(tensor.dot(layer) + bias, model=activation)
        
        else:

            return self.activate(tensor.dot(layer), model=activation)
    
    def propagateLayer(self, layer_out, Input): # works

        c = self.activate(Input, model=self.activation_overlay[0])

        for layer in range(1, layer_out + 1):

            c = self.transpose(self.tensors[layer], c, bias=self.biases[layer], activation=self.activation_overlay[layer])

        return c
    
    def propagate(self, Input): # works

        if type(Input) == list:

            Input = np.array(Input)

        return self.propagateLayer(self.L, Input)

        # last_hidden = self.propagateLayer(self.L-1, Input)

        # return self.transpose(self.tensors[self.L], last_hidden, bias=self.biases[self.L], activation=self.activation_overlay[-1]) # the output layer is not squashed

    def activate(self, x, derivative=False, model='sigmoid'): # works

        if model == 'sigmoid':

            sig = lambda x: 1 / ( 1 + np.exp(-x) )

            if derivative:

                return sig(x) * (1 - sig(x))

            return sig(x) - 1/2
        
        elif model == 'relu':

            if type(x) == np.ndarray or type(x) == list:

                out, val = [], None

                for X in x:

                    val = max(0, X)

                    if derivative:
                        
                        if val > 0:
                        
                            out.append(1.)
                        
                        else:

                            out.append(0.)
                    
                    else:

                        out.append(val)

                return np.array(out)
            
            else:

                if derivative:

                    if x > 0:
                        
                        return 1.
                        
                    return 0.

                return max(0, x)

        elif model == 'id':

            if derivative:
                
                if type(x) == np.ndarray or type(x) == list:

                    return np.array([1. for i in x])
                
                return 1.
            
            return x
        
        elif model == 'softmax':

            if derivative:

                # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
                # input s is softmax value of the original input x. 
                # s.shape = (1, n) 
                # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])    # initialize the 2-D jacobian matrix.
                s = np.exp(x) / sum(np.exp(x))
                jacobian_m = np.diag(s)    
                for i in range(len(jacobian_m)):
                    for j in range(len(jacobian_m)):
                        if i == j:
                            jacobian_m[i][j] = s[i] * (1-s[i])
                        else: 
                            jacobian_m[i][j] = -s[i]*s[j]
                return np.linalg.det(jacobian_m)

            z_vec = x
            z_exp = np.exp(z_vec)

            N = sum(z_exp)

            return z_exp / N

    def loss(self, Output, Target, model='chi_squared'): # works
        if model == 'chi_squared':
            return error.chi_squared(Output, Target)

    def nudge(self, layer, i, j=0, change=0, what='weights'):

        if what == 'weights':

            if type(change) in [np.ndarray, list]:
                self.tensors[layer] = self.tensors[layer] + change
            else:
                self.tensors[layer][i][j] = self.tensors[layer][i][j] + change
        
        if what == 'biases':
            
            if type(change) in [np.ndarray, list]:
                self.biases[layer] = self.biases[layer] + change
            else:
                self.biases[layer][i] = self.biases[layer][i] + change

    def backprop(self, Input, Target, nudge=False):

        '''
        nudge: if True the change will be applied
        '''

        # quick Output
        Output = self.propagate(Input)

        overlays = {}
        deltas = []
        bias_overlays = {}

        # iterate from L to 2
        for layer in range(self.L, 0, -1):

            # define demanded layers and tensors
            if layer == 1:
                layer_prior = self.activate(Input, model=self.activation_overlay[0])
            else:
                layer_prior = self.propagateLayer(layer-1, Input)
            tensor_current = self.tensors[layer]
            
            changeOverlay = np.zeros( shape=( tensor_current.shape[0], tensor_current[0].shape[0] ) )
            bias_overlay = np.zeros( shape=(tensor_current.shape[0]) )

            if layer == self.L:

                layer_current = self.transpose(tensor_current, layer_prior, bias=self.biases[layer], activation='id')
                d = 2 * (layer_current - Target)
                
            else:

                layer_current = self.transpose(tensor_current, layer_prior, bias=self.biases[layer], activation=self.activation_overlay[layer-1])
                tensor_post = self.tensors[layer+1]
                layer_post = self.transpose(tensor_post, layer_current, self.biases[layer+1], activation='id')
                d = []

                for k in range(tensor_post.shape[1]):

                    new = 0
                    
                    for j in range(tensor_post.shape[0]):
                    
                        new += tensor_post[j][k] * self.activate(layer_post[j], True, self.activation_overlay[layer-1]) * deltas[-1][j]
                    
                    d.append(new)
                
                d = np.array(d)

            deltas.append(d)

            # compute derivative of activation function of the weighted sum of layer l
            sigma_prime = self.activate(layer_current, True, model=self.activation_overlay[layer-1])
            learn = self.learningRate
            
            for i in range(tensor_current.shape[0]):

                for j in range(tensor_current[0].shape[0]):

                    change = (-1) * learn * ( (1 - self.momentum) * ( layer_prior[j] * sigma_prime[i] * deltas[-1][i] ) + self.momentum * self.changes[layer][i][j] )
                    self.changes[layer][i][j] = change
                    changeOverlay[i][j] = change

                bias_overlay[i] = - ( sigma_prime[i] * deltas[-1][i] * learn )

            bias_overlays[layer] = bias_overlay
            overlays[layer] = changeOverlay

        # propagation error/loss
        loss = self.loss(Output, Target)

        # apply changes (vectorized)
        for layer in range(self.L, 0, -1):

            self.tensors[layer] = self.tensors[layer] + overlays[layer]
            self.biases[layer] = self.biases[layer] + bias_overlays[layer]

        # --- Return demanded --- #
        return {'weights': overlays, 'biases': bias_overlays, 'loss': loss}

    def train(self, sample, batchSize=1, epochs=1, verbose=False, error_plot=False, stop=0.0, learning_decay=0.0, learning_rate=None, shuffle=True):

        '''
        batchSize: if 1 then its stochastic gradient decent, if its equal len(sample) its batch g. d., everything else is mini-batch
        learning_rate: if 0.0 then the predefined learning rate will be taken else it will be overwritten
        '''

        # print all parameters

        for i in range(5, 0, -1):
            if verbose:
                print('[train] training will start in', i, end='\r')
            time.sleep(1.0)
        if verbose:
            print('start                            ', end='\r')

        # def learning rate and decay
        if learning_rate != None:
            self.learningRate = learning_rate
        learning = self.learningRate
        self.learning_decay = learning_decay

        # error
        error = []

        # iterate epochs
        for epoch in range(epochs):

            if verbose:
                print('\n[train verbose]: epoch:', epoch+1, '\n')

            # first shuffle if demanded
            if shuffle:
                np.random.shuffle(sample)

            # apply batch size and break up the sample
            self.batchSize = batchSize
            batches, i = [], 0
            while True:
                snippet = sample[ i * batchSize : (i+1) * batchSize - 1]
                if snippet == []:
                    break
                batches.append(snippet)
                i += 1

            # iterate through batches
            for b in range(len(batches)):

                # remember all weight and bias change overlays
                biasOverlays = []
                weightOverlays = []
                losses = []
                
                # iterate through batch where each one is an (input, target)-tuple
                for i, t in batches[b]:
                    
                    bp = self.backprop(i, t)
                    biasOverlays.append(bp['biases'])
                    weightOverlays.append(bp['weights'])
                    losses.append(bp['loss'])

                # compute the loss
                self.error = np.mean(np.array(losses))
                error.append(self.error)

                # --- compute the mean change --- #

                for layer in range(1, self.L + 1):

                    # average the desired changes
                    bias_change_overlay_mean = sum([ol[layer] for ol in biasOverlays]) / self.batchSize     # vector
                    weight_change_overlay_mean = sum([ol[layer] for ol in weightOverlays]) / self.batchSize # matrix
                    
                    for i in range(weight_change_overlay_mean.shape[0]):

                        for j in range(weight_change_overlay_mean[0].shape[0]):

                            self.nudge(layer, i, j, weight_change_overlay_mean[i][j], 'weights')

                        self.nudge(layer, i, change=weight_change_overlay_mean[i][j], what='biases')
                

                # verbose information
                if verbose:
                    print('[train verbose]: batch = {}; cost = {}'.format(b, self.error))

                # stop
                if self.error <= stop and stop != 0:
                    print('[train]: stopping. Current loss {} has undergone the threshold.'.format(self.error))
                    break

                # update learning rate
                if learning_decay != 0.0:
                    self.learningRate = learning / ( 1.0 + learning_decay * (batchSize*epoch + b) )
            
            if verbose:
                print('[evaluate random]: output = {}; desired = {}'.format(self.propagate(sample[0][0]), sample[0][1]))

        if error_plot:

            plt.plot(error)
            plt.grid()
            plt.show()

        if verbose:
            print('[train verbose]: Done.')
        
        return 0