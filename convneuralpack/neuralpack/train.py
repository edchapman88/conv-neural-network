from neuralpack.model import SerialModel
from typing import Callable, Iterator
import numpy as np
import logging

logger  = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class Trainer:
    def __init__(self, model:SerialModel, sample_generator_func:Callable):
        self.model = model
        self.sample_generator = sample_generator_func
        self.sample_itr = sample_generator_func()
        

    def run(self, batch_size=1, epochs=1, learning_rate=0.1):
        self.batch_itr = self.batch_generator(batch_size,self.sample_itr)
        batch_errors = []
        
        for epoch in range(1, epochs+1):
            epoch_errors = []
            while True:
                try:
                    X,Y = next(self.batch_itr)  # get next batch in this epoch

                except StopIteration:   # end of epoch when sample generator reaches stopIteration
                    # reset generators for next epoch
                    self.sample_itr = self.sample_generator()
                    self.batch_itr = self.batch_generator(batch_size,self.sample_itr)
                    break # out of while loop for this epoch
                
                # train model after each batch
                batch_error = self.model.train(batch_X=X, batch_Y=Y, learning_rate=learning_rate)
                batch_errors.append(batch_error)
                epoch_errors.append(batch_error)

            logger.info(f'Epoch {epoch} complete with mean error: {sum(epoch_errors)/len(epoch_errors)} ')

        return batch_errors

    def batch_generator(self, batch_size, sample_itr):
        more_data = True
        while more_data == True:
            X = []
            Y = []

            for i in range(batch_size):
                try:
                    x,y = next(sample_itr)
                except StopIteration:
                    more_data = False
                    break
                X.append(x)
                Y.append(y)
            
            if more_data:
                X = np.reshape(X, (batch_size,2,1))
                Y = np.reshape(Y, (batch_size,1,1))
                yield X,Y