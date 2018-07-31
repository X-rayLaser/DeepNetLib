Feature: Testing a vanilla neural network


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
