
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
2. activation overlay -  a python list with same dimension L with the corresponding activation function
3. a name (optional) 

With these parameters build a new pyron network by calling the Object `pyron.network` and initialize with the `pyron.network.new` method
```python
from pyron import network

net = network()
net.new(
    name='myNewANN', 
    topology=[10, 8, 8, 4], 
    activation_overlay=['id', 'relu', 'sigmoid', 'softmax'] )
```
The net was initialized with 4 layers including the input and output layers. By this instance the input must be an array of dimension 10 and is then propagated via two 8-dimensional layers and outputted in a 4 dimensional one i.e. the output is an ndarray of dimension (0, 4). Each layer 
obtains an appropriate activation function with which the output of each layer is convolved.
The possible activation functions are

 - 'id' : identical function (no convolution)
 - 'relu': relu activation function
 - 'sigmoid': sigmoid function
 - 'softmax': softmax function (preserves probability)
  
<br>

### training
pyron is trained with labeled data (supervised learning) and minimizes the loss function by a simple [batch, mini-batch or stochastic gradient decent](https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20first,the%20direction%20of%20steepest%20descent.) backpropagation algorithm. The training sample must be packaged accordingly

```python
sample = [[inputArray1, targetArray1], [inputArray2, targetArray2], ... ] 
```

recall that `inputArray` and `targetArray` must be ndarrays of dimension 10 and 4, respectively. Passing the data is now straightforward with the `pyron.network.train` method

```python
net.train(sample)
```

`network.train()` takes the following arguments
 - <strong>sample</strong> : array of input and target arrays
 - <strong>batchSize</strong>: number of (input, target)-pairs per propagated sample slice before backprop.
 - <strong>epochs</strong>: number of total sample propagations
 - <strong>verbose</strong>: for verbose printing during training
 - <strong>error_plot</strong>: plots the loss function (mean-squared-error) over time when training is finished
 - <strong>stop</strong>: loss function value after which to stop the training (default=0.0)
 - <strong>learning_decay</strong>: learning rate decay exponential factor (default=0.0)
 - <strong>learning_rate</strong>: learning rate (default=0.01)
 - <strong>shuffle</strong>: shuffle the sample before every epoch (default=True)

<strong>Note</strong>: Many usecases in machine learning demand unique hyper parameters and it is crucial to pre-process the data like e.g. normalization or min-max-scaling, which will improve the learning efficiency significantly. The tuning of the parameters might take a few trainings so make sure to have the `error_plot` set to true to see the learning/loss curve after each training and to avoid overfitting.

<br>

### testing
When a suitable loss is reached the freshly trained perceptron requires some test data to be propagated through. In order to mimick a real testsuite it is recommended to take `inputArrays` indifferent from the training sample. For a single forward propagation of one (input, target) pair the syntax might look like

```python
# test data sample
inputArray = [x1, x2, ... , x10]
targetArray = [y1, y2, ... , y4]

# get a perceptron prediction 
output = net.propagate(inputArray)

# error of prediction
```
