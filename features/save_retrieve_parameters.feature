Feature: Saving and retrieving weights and biases to and from a file


@persist @quadratic
Scenario: Train network, save its parameters in a file and initialize new net with them
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I randomly initialize net's parameters
    And I choose quadratic cost function
    And I choose to use gradient descent as learning algorithm
    And I train neural network on that data set for 10 epochs
    And I memorize weights and biases
    And I save neural net weights and biases to file "neural_net.json"
    And I initialize neural net parameters from a file "neural_net.json"
    Then new net parameters match the parameters of old neural net
