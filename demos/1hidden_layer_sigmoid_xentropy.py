import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from demos.mnist_trainer import MnistTrainer, TrainerConfig
from cost_functions import CrossEntropyCost
from activation_functions import Sigmoid


conf = TrainerConfig.make_config(hidden_layer_sizes=[60],
                                 hidden_activation=Sigmoid,
                                 output_activation=Sigmoid,
                                 loss_function=CrossEntropyCost,
                                 L2_regularization_term=0,
                                 mini_batch_size=10)

trainer = MnistTrainer(conf)
trainer.start(nepoch=10)
