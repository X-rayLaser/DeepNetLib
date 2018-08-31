Feature: Testing a back propagation algorithm correctness

@backprop @quadratic
Scenario: Back propagation outputs correct gradients when using quadratic cost
    When I initialize simple neural net with default parameters
    And I randomly initialize net's parameters
    And I choose quadratic cost function
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

@backprop @quadratic @rectifier
Scenario: Back propagation outputs correct gradients when using rectifier activation function
    When I initialize simple neural net with default parameters
    And I randomly initialize net's parameters
    And I choose rectifier activation function
    And I choose quadratic cost function
    And I generate a data set from a function "sin(x)^2"
    And I compute the gradient for weights and biases by running back propagation
    And I compute the gradient for weights and biases by taking numerical derivatives
    Then these two sets of gradients are the same

@backprop @cross_entropy @softmax
Scenario: Back propagation outputs correct gradients when using linear hidden units with softmax activation in final layer
    When I initialize simple neural net with default parameters
    And I randomly initialize net's parameters
    And I choose rectifier activation function for hidden layer(s)
    And I choose softmax activation function for output layer
    And I choose cross entropy cost function
    And I generate a data set from a function "sin(x)^2"
    And I compute the gradient for weights and biases by running back propagation
    And I compute the gradient for weights and biases by taking numerical derivatives
    Then these two sets of gradients are the same
