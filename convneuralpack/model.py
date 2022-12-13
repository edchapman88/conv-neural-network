import numpy as np
import logging
import math
from typing import NewType,Literal


logger = logging.getLogger(__name__)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)

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

    def train(self, batch_X:np.ndarray, batch_Y:np.ndarray, learning_rate:float, loss_fn):
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

        if loss_fn == 'MSE':
            loss_fn = mse
            loss_fn_prime = mse_prime
        elif loss_fn == 'BCE':
            loss_fn = bin_cross_entropy
            loss_fn_prime = bin_cross_entropy_prime
        else:
            logger.error(f'Loss function "{loss_fn}", not recognised.')

        error_sum = 0
        error_grad_sum = 0
        for i in range(batch_X.shape[0]):
            x = batch_X[i]
            y = batch_Y[i]
            y_pred = self.predict(x)
            error = loss_fn(y_true=y, y_pred=y_pred)
            error_sum += error
            error_grad = loss_fn_prime(y_true=y,y_pred=y_pred)
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
        # use He normal initializer (optimised for ReLu activation layers)
        std = math.sqrt(2 / input_size)
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

def bin_cross_entropy(y_true,y_pred):
    # one = np.log(y_pred)
    # print(f'one: {one}')
    # two = y_true * np.log(y_pred)
    # print(f'two: {two}')
    # three = np.log(1 - y_pred)
    # print(f'three: {three}')
    # four = (1 - y_true) * np.log(1 - y_pred)
    # print(f'four: {four}')

    for idx in range(len(y_pred)):
        if y_pred[idx] == 1:
            y_pred[idx] -= 1e-20


    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bin_cross_entropy_prime(y_true, y_pred):
    return ((1-y_true) / (1-y_pred) - y_true/y_pred) / np.size(y_true)


class ReshapeLayer(BaseLayer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, e_grad_out, learning_rate):
        return np.reshape(e_grad_out, self.input_shape)

class ConvLayer(BaseLayer):
    def __init__(self,input_shape,kernel_size,depth):
        self.input_shape = input_shape
        input_depth,input_height,input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_shape = (depth,input_height - kernel_size + 1, input_width - kernel_size + 1)
        # matrix of all kernerls (4th dimension across individual kernels, 3rd dimension is because there
        # is a seperate kernel for each layer in the input (eg. RGB layers)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # assumes kernel is square
        
        # use He normal initializer (optimised for ReLu activation layers)
        std = math.sqrt(2 / input_depth)
        self.weights = np_rng.normal(loc=0, scale=std, size=self.kernels_shape)
        self.bias = np.zeros(shape=self.output_shape)

    def forward(self, input):
        # save input tensor to be used in backpropagation
        self.input = input
        # loop through depth of this layer (eg. number of kernels)
        output = np.zeros(self.output_shape)
        for kernel_idx in range(self.depth):
            for input_layer_idx in range(self.input_shape[0]):
                output[kernel_idx,:,:] += self.valid_cross_correlation(input[input_layer_idx,:,:],self.weights[kernel_idx,input_layer_idx,:,:])
            output[kernel_idx,:,:] += self.bias[kernel_idx,:,:]
        return output

    def backward(self, e_grad_out, learning_rate):
        weights_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)
        for kernel_idx in range(self.depth):
            for input_layer_idx in range(self.input_shape[0]):
                weights_grad[kernel_idx,input_layer_idx,:,:] = self.valid_cross_correlation(self.input[input_layer_idx],e_grad_out[kernel_idx])
                input_grad[input_layer_idx] += self.full_convolution(e_grad_out[kernel_idx],self.weights[kernel_idx,input_layer_idx])
        
        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * e_grad_out
        return input_grad

    def valid_cross_correlation(self,matrix,kernel):
        output_shape = (matrix.shape[0] - kernel.shape[0] + 1, matrix.shape[1] - kernel.shape[1] + 1)
        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]
        out = np.zeros(output_shape)
        for j in range(output_shape[0]):
            for i in range(output_shape[1]):
                # elementwise multiplication
                product = matrix[j:j+kernel_height,i:i+kernel_width] * kernel
                out[j,i] = product.sum()

        return out

    def full_convolution(self,matrix,kernel):
        rotated_kernel = self.rotate_180(kernel)

        # padding on each side
        padding_j = kernel.shape[0] - 1
        padding_i = kernel.shape[1] - 1
        padded = np.pad(matrix, ((padding_j,padding_j),(padding_i,padding_i)), constant_values=0)

        out = self.valid_cross_correlation(padded, rotated_kernel)
        return out
    
    def rotate_180(self,matrix):
        out = np.zeros(matrix.shape)
        (J,I) = matrix.shape
        for j in range(J):
            for i in range(I):
                out[j,i] = matrix[J-1-j,I-1-i]
        return out
    
class SoftmaxLayer(BaseLayer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        for num in self.output:
            if num == 0:
                print('found a 0!')
        return self.output
    
    def backward(self, e_grad_out, learning_rate):
        n = np.size(self.output)
        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), e_grad_out)