import numpy as np
from scipy.signal import correlate2d, convolve2d

def valid_cross_correlation(matrix,kernel):
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

def rotate_180(matrix):
    out = np.zeros(matrix.shape)
    (J,I) = matrix.shape
    for j in range(J):
        for i in range(I):
            out[j,i] = matrix[J-1-j,I-1-i]
    return out

def full_convolution(matrix,kernel):
        rotated_kernel = rotate_180(kernel)

        # padding on each side
        padding_j = kernel.shape[0] - 1
        padding_i = kernel.shape[1] - 1
        padded = np.pad(matrix, ((padding_j,padding_j),(padding_i,padding_i)), constant_values=0)

        out = valid_cross_correlation(padded, rotated_kernel)
        return out

np_rng = np.random.default_rng()
ones = np_rng.normal(loc=0, size=(3,3))

filter = np_rng.normal(size=(2,2))

full_convolution(ones,filter)

# print(convolve2d(ones,filter,'full'))

