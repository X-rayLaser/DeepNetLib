Feature: Testing a neural network is able to generalize for previously unseen examples


@general @quadratic
Scenario: Net can guess a distribution from which examples are drawn
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I randomly initialize net's parameters
    And I choose quadratic cost function
    And I choose to use gradient descent as learning algorithm
    And I train neural network on that data set for 100 epochs
    Then neural net gives less than 40% classification error on test data set

@general @cross_entropy
Scenario: Net can guess a distribution from which examples are drawn
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I randomly initialize net's parameters
    And I choose cross entropy cost function
    And I choose to use gradient descent as learning algorithm
    And I train neural network on that data set for 100 epochs
    Then neural net gives less than 40% classification error on test data set

@general @quadratic @stochastic
Scenario: Net is trained to infer distribution generating x using stochastic gradient descent
    When I generate a data set consisting of sequences of length 10 drawn from different distributions
    And I initialize a neural net for binary classification with sizes 10,5,3
    And I choose quadratic cost function
    And I choose stochastic gradient descent as a learning algorithm with learning rate 3.0
    And I randomly initialize net's parameters
    And I train neural network on that data set for 100 epochs
    Then neural net gives less than 40% classification error on test data set

@general @cross_entropy @stochastic @mnist @long
Scenario: Net is trained to classify MNIST hand-written digits
    When I create a training and testing data from MNIST data set
    And I initialize a neural net for binary classification with sizes 784,30,10
    And I choose cross entropy cost function
    And I choose stochastic gradient descent as a learning algorithm with learning rate 0.1
    And I randomly initialize net's parameters
    And I train neural network on that data set for 1 epochs
    Then neural net gives less than 50% classification error on test data set
