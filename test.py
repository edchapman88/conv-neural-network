from convneuralpack.model import ConvLayer,ReshapeLayer,SerialModel,ReluLayer,DenseLayer,SoftmaxLayer
from convneuralpack.train import Trainer

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = mnist.load_data()

def preprocess(x,y,limit):
    zero_index = np.where(y==0)[0][:limit]
    one_index = np.where(y==1)[0][:limit]
    all_indicies = np.hstack((zero_index, one_index))
    all_indicies = np.random.permutation(all_indicies)
    x,y = x[all_indicies], y[all_indicies]
    x = x.reshape(len(x),1,28,28)
    x = x.astype('float32') /255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y),2,1)
    return x,y

x_train,y_train = preprocess(x_train,y_train,100)
x_test,y_test = preprocess(x_test,y_test,100)

print(x_train.shape)
print(y_train.shape)


model = SerialModel([
    ConvLayer(input_shape=(1,28,28), kernel_size=(3), depth=5),
    ReluLayer(),
    ReshapeLayer(input_shape=(5,26,26), output_shape=(5*26*26,1)),
    DenseLayer(input_size=(5*26*26), output_size=100),
    ReluLayer(),
    DenseLayer(input_size=100, output_size=2),
    SoftmaxLayer()
])


# model.train(np.reshape(x_test[0],(1,1,28,28)),np.reshape(y_test[0],(1,2,1)),learning_rate=0.1,loss_fn='BCE')
print(model.predict(x_train[0]))

trainer = Trainer(model, dataset=(x_train,y_train))

batch_errors = trainer.run(batch_size=1, epochs=1, learning_rate=0.01, loss_fn='BCE')

preds = []
correct = 0
for idx in range(x_test.shape[0]):
    pred = model.predict(x_test[idx])
    preds.append(pred)
    if np.argmax(pred) == np.argmax(y_test[idx]):
        correct += 1

print(f'accuracy = {correct/x_test.shape[0]}')
# print(preds)

plt.plot(batch_errors)
plt.show()

