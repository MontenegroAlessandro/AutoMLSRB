# Libraries
from sklearn.neural_network import MLPClassifier


# Wrapper of MLPClassifier making possible to tune the network shape
class MyMLPClassifier(MLPClassifier):
    def __init__(self, hidden_layer_size, hidden_layer_number, activation, solver, alpha, learning_rate,
                 learning_rate_init, power_t,max_iter, tol, momentum, beta_1):
        # define the tuple with the desired network shape
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_number = hidden_layer_number
        self.hidden_layer_sizes = hidden_layer_number * (hidden_layer_size,)

        # initialize the MLPClassifier
        super().__init__(hidden_layer_sizes=self.hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                         learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t,
                         max_iter=max_iter, tol=tol, momentum=momentum, beta_1=beta_1)
