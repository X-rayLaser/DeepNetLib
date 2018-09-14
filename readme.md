## Welcome!
## Getting started

Open a terminal

Install virtualenv.

    pip install virtualenv

Create a new virtual environment for python.

    virtualenv venv
Alternatively, if you have multiple versions of Python on your system you can choose which one to use.

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



## Running tests
Run unit and integration tests.

    python tests.py
Run functional tests

    behave

 All tests should pass, but if one or 2 fail on rare occasion,  than it is ok. There is a certain randomness intrinsic to the training process causing 1 or 2 tests checking classification accuracy to fail sometimes. A few  warnings are also ok.
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