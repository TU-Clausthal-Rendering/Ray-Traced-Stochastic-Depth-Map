# config
dataPath = 'D:/VAO/'

# imports
import os
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D

# set current directory as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def write_weights(filename, model):
    weights = model.get_weights()
    for i in range(len(weights)):
        type = "weight"
        if i % 2 == 1:
            type = "bias"
        
        # save as numpy array
        np.save(f'{filename}{type}_{i//2}.npy', weights[i])

# load the model
#model = keras.models.load_model('model_eval8_2_relu.h5')
#write_weights('', model)

for slice in range(16):
    model = keras.models.load_model(f'model{slice}_eval.h5')
    write_weights(f'{slice}_', model)