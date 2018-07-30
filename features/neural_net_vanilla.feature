Feature: Testing a vanilla neural network


Scenario: Trained network approximates sin(x)^2 math function
    When I initialize simple neural net with default parameters
    And I train neural network to approximate a function "sin(x)^2"
    Then I feed it a set of inputs and get correct results
