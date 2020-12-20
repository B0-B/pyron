
# pyron

pyron was developed to quickly create and train an artificial neural network from scratch without the need of large third-party installations which by empiricism lead to an overkill especially on single-board hosts with simple tasks.

## features
- easy to set up
- automatic plotting
- perfect for little tasks
- dump in json

## implementation
### dependencies
- [numpy](https://numpy.org/) for better computational performance
  
After cloning the repository prompt in the root directory
```
~pyron$ pip install .
```

## usage

<br>

### initialization


A perceptron network with L hidden layers can be reduced to a topology like
```
layer(l): 0         1           2        L-1         L

        input   >   °     >     °         °     >   Out
        input   >   °     >     °   ...   °     >   Out
        input   >   °     >     °         °     >   Out
        input   >   °     >     °         °     >   Out
```
where the arrows symbolize the tensors which are build during initialization. The init for the raw shape demands some parameters 

1. layer topology L and neuron number for each layer which is achieved by a single python list of dimension L+1 (the 1 is the input layer).
2. activation overlay -  a python list with same dimension L with the appropriate activation function
3. a name (optional) 

With this parameters build a new pyron network by calling the Object `pyron.network` and initialize with the `pyron.network.new` method
```python
from pyron import network

net = network()
net.new(
    name='myNewANN', 
    topology=[10, 8, 8, 4], 
    activation_overlay=['id', 'relu', 'sigmoid', 'softmax'] )
```
and was initialized with 4 layers including the input and output layers. By this instance the input must be an array of dimension 10 and is then propagated via two 8-dimensional layers and outputted in a 4 dimensional one i.e. the output is an ndarray of dimension (0, 4). Each layer 
obtains an appropriate activation function with which the output of each layer is convolved.
The possible activation functions are

 - 'id' : identical function (no convolution)
 - 'relu': relu activation function
 - 'sigmoid': sigmoid function
 - 'softmax': softmax function (preserves probability)
  
<br>

### training
pyron is trained with labeled data (supervised learning) and minimizes the loss function by a simple [batch, mini-batch or stochastic gradient decent](https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20first,the%20direction%20of%20steepest%20descent.) backpropagation algorithm. The training sample 