import numpy as np
import math
from typing import Callable
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

np_rng = np.random.default_rng()

class SerialModel:
    def __init__(self, layers:list):
        self.layers = layers

    def predict(self, X:np.ndarray):
        input = X
        for layer in self.layers:
            output = layer.forward(input=input)
            input = output
        return output

    def train(self, batch_X:np.ndarray, batch_Y:np.ndarray, learning_rate:float):
        '''
        Train the network on a batch of data.
        
        Passing a batch consisting of one data
        point is equivlent to stocastic gradient descent. Otherwise "mini-batch"
        gradient descent is used. The network weights are updated once with a mean
        error gradient calculated across the batch.

        PARAMETERS:
        batch_X (np.ndarray(shape[batch_size, num_model_inputs, 1])): A batch of model inputs.
        batch_Y (np.ndarray(shape[batch_size, num_model_outputs, 1])): A batch of expected model outputs
        learning_rate (float): Step size during gradient descent.

        RETURNS:
        mean_batch_error (float): Mean Squared Error calculated across the batch during training.
        '''

        error_sum = 0
        error_grad_sum = 0
        for i in range(batch_X.shape[0]):
            x = batch_X[i,:,:]
            y = batch_Y[i,:]
            y_pred = self.predict(x)
            error = mse(y_true=y, y_pred=y_pred)
            error_sum += error
            error_grad = mse_prime(y_true=y,y_pred=y_pred)
            error_grad_sum += error_grad

        mean_batch_error = error_sum / batch_X.shape[0]
        # mean_batch_error_grad 
        e_grad_out = error_grad_sum / batch_X.shape[0]

        for layer in reversed(self.layers):
            e_grad_in = layer.backward(e_grad_out=e_grad_out, learning_rate=learning_rate)
            e_grad_out = e_grad_in

        return mean_batch_error
        

            

class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self,input):
        self.input = input

    def backward(self,e_grad_out,learning_rate):
        pass


class DenseLayer(BaseLayer):
    def __init__(self,input_size,output_size):
        # use Xavier normal initializer
        std = math.sqrt(2 / (input_size + output_size))
        self.weights = np_rng.normal(loc=0, scale=std, size=(output_size,input_size))
        self.bias = np.zeros(shape=(output_size,1))

    def forward(self, input):
        # Y = W.X + B
        # shapes (j,) = (j,i).(i,) + (j,)
        self.input = input
        return np.dot(self.weights,input) + self.bias

    def backward(self, e_grad_out, learning_rate):
        '''
        Matrix Algebra:

        dE/dW = dE/dY . transpose(X)

        dE/DB = dE/dY

        dE/dX = transpose(W) . dE/dY
        '''
        e_grad_weights = np.dot(e_grad_out,np.transpose(self.input))
        e_grad_bias = e_grad_out
        e_grad_in = np.dot(np.transpose(self.weights),e_grad_out)
        self.weights -= learning_rate * e_grad_weights
        self.bias -= learning_rate * e_grad_bias
        return e_grad_in

class ActivationLayer(BaseLayer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, e_grad_out, learning_rate):
        '''
        dE/dW = dE/dY .(elemwise) f_prime(X)
        '''
        return np.multiply(e_grad_out,self.activation_prime(self.input))

class TanhLayer(ActivationLayer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class ReluLayer(ActivationLayer):
    def __init__(self):
        relu = lambda x: np.maximum(x,np.zeros_like(x))
        relu_prime = lambda x: np.where(x<=0,0,1)
        super().__init__(relu, relu_prime)

def mse(y_true,y_pred):
    return np.mean(np.power(np.subtract(y_true,y_pred),2))

def mse_prime(y_true,y_pred):
    return 2 * np.subtract(y_pred,y_true) / np.size(y_true)
        

