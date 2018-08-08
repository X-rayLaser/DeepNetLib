Feature: Checking images of hand-written digits produced by neural net


@generating @long
Scenario: Train a net to generate images of digits, run images though digit recognizer net
    When I create a training and testing data from MNIST data set
    And I initialize a neural net for binary classification with sizes 784,30,10
    And I choose stochastic gradient descent as a learning algorithm
    And I choose cross entropy cost function
    And I randomly initialize net's parameters
    And I train neural network on that data set for 1 epochs
    And I train a digit generator
    Then out of 1000 generated digits 50% or more are indeed digits
