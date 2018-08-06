Feature: Testing a vanilla neural network

@backprop
Scenario: Back propagation outputs correct gradients
    When I initialize simple neural net with default parameters
    And I randomly initialize net's parameters
    And I generate a data set from a function "sin(x)^2"
    And I compute the gradient for weights and biases by running back propagation
    And I compute the gradient for weights and biases by taking numerical derivatives
    Then these two sets of gradients are the same

@backprop @cross_entropy
Scenario: Back propagation outputs correct gradients when using cross entropy cost
    When I initialize simple neural net with default parameters
    And I randomly initialize net's parameters
    And I choose cross entropy cost function
    And I generate a data set from a function "sin(x)^2"
    And I compute the gradient for weights and biases by running back propagation
    And I compute the gradient for weights and biases by taking numerical derivatives
    Then these two sets of gradients are the same

Scenario: Convergence of network's parameters when training to approximate sin(x)^2 math function
    When I initialize simple neural net with default parameters
    And I generate a data set from a function "sin(x)^2"
    And I remember initial cost value for that data set
    And I train neural network on that data set for 1000 epochs
    Then the cost function gives much smaller error value than before

Scenario: Trained network approximates sin(x)^2 math function
    When I initialize simple neural net with default parameters
    And I train neural network to approximate a function "sin(x)^2"
    Then I feed it a set of inputs and get correct results

@rand_distrib
Scenario: Net can guess a distribution from which examples are drawn
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I randomly initialize net's parameters
    And I train neural network on that data set for 100 epochs
    Then neural net gives less than 25% classification error on test data set

@cross_entropy
Scenario: Net can guess a distribution from which examples are drawn
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I randomly initialize net's parameters
    And I choose cross entropy cost function
    And I train neural network on that data set for 100 epochs
    Then neural net gives less than 25% classification error on test data set

@cross_entropy
Scenario: Convergence of network's parameters when training using cross entropy cost function
    When I initialize simple neural net with default parameters
    And I choose cross entropy cost function
    And I generate a data set from a function "sin(x)^2"
    And I remember initial cost value for that data set
    And I train neural network on that data set for 1000 epochs
    Then the cost function gives much smaller error value than before

@stochastic
Scenario: Net is trained to infer distribution generating x using stochastic gradient descent
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I choose stochastic gradient descent as a learning algorithm
    And I randomly initialize net's parameters
    And I train neural network on that data set for 100 epochs
    Then neural net gives less than 25% classification error on test data set
