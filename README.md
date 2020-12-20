
# pyron

pyron is a simple python package which was developed to implement and train a neural network from scratch without the need
of large third-party installations which by empiricism lead to an overkill especially on single-board hosts that demand only simple tasks.

## features
- easy to set up
- automatic plotting
- perfect for little tasks
- dump in json

## implementation
### dependencies
- numpy
  
After cloning the repository prompt in the root directory
```
~pyron$ pip install .
```

## usage
A perceptron network with $L$ layers can be reduced to a topology like
```
layer(l): 0         1           2        L-1         L

        input   >   °     >     °         °     >   Out
        input   >   °     >     °   ...   °     >   Out
        input   >   °     >     °         °     >   Out
        input   >   °     >     °         °     >   Out
```
where the arrows symbolize the tensors which are build during initialization. The init for the raw shape demands some parameters 

1. layer topology $L$ and neuron number for each layer which is achieved by a single python list.
2. activation overlay -  a python list with same dimension $L$ with the appropriate activation function
3. a name (optional) 
```
~pyron$ pip install .
```