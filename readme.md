## Welcome!
## Getting started

Open a terminal

Install virtualenv.

    pip install virtualenv

Create a new virtual environment for python.

    virtualenv venv
Alternatively, if you have multiple versions of Python on your system, you can choose which one to use.

    virtualenv venv --python=/full/path/to/python/executable

Activate a virtual environment that you just created.
If you are using windows

    venv\scripts\activate
If you are using linux

    source venv/bin/activate

Clone repository and cd into it.

    git clone https://github
Install projects dependencies

    pip install -r requirements.txt
Now you are all set.

## Simple usage

In this simple example we are going to create a neural network and train it to fit a simple curve y = x*x.
First, let's import everything that we will need

    import numpy as np
    from data_source import PreloadSource
    from neural_net import NetFactory
    from activation_functions import Rectifier, Sigmoid
    from cost_functions import QuadraticCost
    from gradient_descent import GradientDescent

We create a feed-forward neural network with 3 layers with 1 unit in the input layer, 1 unit in the output layer and 200 units in the hidden layer. We use logistic sigmoid for hidden layer as an activation, also we use a Rectifier (or Rectified Linear Unit) as an activation in the output layer.

    net = NetFactory.create_neural_net([1, 200, 1],
                                   hidden_layer_activation=Sigmoid,
                                   output_layer_activation=Rectifier)

An important step is to perform random initialization of weights for symmetry breaking

    net.randomize_parameters()

We are going to optimize quadratic loss function and use L2 regularization

    cost = QuadraticCost(neural_net=net, l2_reg_term=0.0005)

Now we will instantiate a gradient descent object which will try to minimize this loss. The learning_rate parameter indicates how large will be each update step

    gd = GradientDescent(neural_net=net, cost_function=cost,
                         learning_rate=0.001)

Almost there. Now lets generate a few training examples representing a curve y = x*x.

    def x_to_y(x):
        return np.array([x ** 2], float)


    X = []
    Y = []
    x = 2
    while x < 11:
        X.append(np.array([x], float))
        Y.append(x_to_y(x))
        x += 1

    examples = (X, Y)
    src = PreloadSource(examples=examples)

And now we use these examples to train the net for 1000 training epochs

    for i in range(5000):
	    gd.training_epoch(data_src=src)
	    print('Epoch {}. Loss is {}'.format(i, cost.get_cost(data_src=src)))
Hopefully, everything went well. Let's test our trained neural net on a few examples

    t = 2
    while t < 10:
        x = np.array([t], float)
        y_estimate = net.feed(x)
        print('{} -> {}'.format(x, y_estimate))
        t += 0.1

## Running tests

Run unit and integration tests.

    python tests.py
Run functional tests

    behave

All tests should pass, but if one or 2 fail on rare occasion,  than it is ok. There is a certain randomness intrinsic
to the training process causing 1 or 2 tests checking classification accuracy to fail sometimes.
A few warnings are also ok.

 ## Trying demos

Try out these python scripts performing classification tasks on MNIST data set. You should wait a couple of minutes after launch before you see any feedback text such as current learning iteration and classification accuracy.

Keep in mind that some scripts are expected to run for quite some time before they finish. Feel free to terminate the program at any time by pressing ctrl+c.

**Classification scripts**

One hidden layer, sigmoid activation function for all layers, unregularized cross entropy loss.

    python 1hidden_layer_sigmoid_xentropy.py

2 hidden layers, rectified linear units (RELU) for hidden layers, soft-max activation for output layer. Cross entropy loss with L2 regularization.

    python 2hidden_layers_relu_softmax_xentropy_L2.py

4 hidden layers (RELU) and soft-max activation function in output layer. Cross entropy loss with L2 regularization.

    python 4hidden_layers_relu_softmax_xentropy_L2.py


**Digit generation**

Generate images of hand-written digits from vectors.

    python generating_digits.py

Generate images from vectors corrupted by guassian noise.

    python noisy_digits.py

## License
MIT license.