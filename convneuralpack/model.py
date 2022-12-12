import numpy as np
import logging
import math
from neuralpack.model import BaseLayer

from scipy.signal import correlate2d, convolve2d  # for dev assertion

logger = logging.getLogger(__name__)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)

np_rng = np.random.default_rng()

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
                output[kernel_idx,:,:] += self.valid_cross_correlation(self.weights[kernel_idx,input_layer_idx,:,:],input[input_layer_idx,:,:])
            output[kernel_idx,:,:] += self.bias[kernel_idx,:,:]
        return output

    def backward(self, e_grad_out, learning_rate):
        weights_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeroes(self.input_shape)
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
    