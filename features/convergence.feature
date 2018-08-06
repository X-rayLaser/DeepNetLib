Feature: Checking that neural net converges


@convergence @quadratic
Scenario: Convergence of network's parameters when training to approximate sin(x)^2 math function
    When I initialize simple neural net with default parameters
    And I generate a data set from a function "sin(x)^2"
    And I remember initial cost value for that data set
    And I train neural network on that data set for 1000 epochs
    Then the cost function gives much smaller error value than before

@convergence @cross_entropy
Scenario: Convergence of network's parameters when training using cross entropy cost function
    When I initialize a neural net for binary classification with sizes 10,5,3
    And I choose cross entropy cost function
    And I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I remember initial cost value for that data set
    And I train neural network on that data set for 500 epochs
    Then the cost function gives much smaller error value than before
