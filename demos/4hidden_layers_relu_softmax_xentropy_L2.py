import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from demos.mnist_trainer import MnistTrainer, TrainerConfig
from cost_functions import CrossEntropyCost
from activation_functions import Rectifier, Sigmoid, Softmax


conf = TrainerConfig.make_config(hidden_layer_sizes=[13, 12, 11, 10],
                                 hidden_activation=Rectifier,
                                 output_activation=Softmax,
                                 loss_function=CrossEntropyCost,
                                 L2_regularization_term=0.025,
                                 learning_rate=0.005,
                                 mini_batch_size=50,
                                 dataset_size=None)

trainer = MnistTrainer(conf)
trainer.start(nepoch=100)
