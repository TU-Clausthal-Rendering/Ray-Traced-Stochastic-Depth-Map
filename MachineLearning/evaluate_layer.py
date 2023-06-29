# config
dataPath = 'D:/VAO/valid_'
modelName = 'layer2'
#modelName = 'eval8_2_relu'
sample_id = 0

# imports
import os
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from shared import *

# set current directory as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# returns true if sample was processed, false if sample was not processed because it does not exist
def process_sample(model : keras.models.Model, slice):
    # check if f'{dataPath}bright_{sample_id}.npy' exists
    if not os.path.isfile(f'{dataPath}bright_{sample_id}.npy'):
        raise Exception(f'{dataPath}bright_{sample_id}.npy does not exist')

    # Load and preprocess input images
    image_bright = np.load(f'{dataPath}bright_{sample_id}.npy')
    image_dark = np.load(f'{dataPath}dark_{sample_id}.npy')
    image_depth = np.load(f'{dataPath}depth_{sample_id}.npy')
    #image_invDepth = np.load(f'{dataPath}invDepth_{sample_id}.npy')

    # arrays have uint values 0 - 255. Convert to floats 0.0 - 1.0
    image_bright = image_bright.astype(np.float32) / 255.0
    image_dark = image_dark.astype(np.float32) / 255.0
    image_importance = np.ones(image_bright.shape, dtype=np.float32) - (image_bright - image_dark) # importance = bright - dark
    #image_importance = np.zeros(image_bright.shape, dtype=np.float32) # importance = 0

    # Preprocess images and expand dimensions
    #input_data = np.expand_dims(np.concatenate((image_bright, image_dark), axis=2), axis=0)
    #input_data = [np.expand_dims(image_bright, axis=0), np.expand_dims(image_dark, axis=0), np.expand_dims(image_importance, axis=0), np.expand_dims(image_invDepth, axis=0)]
    input_data = [image_bright, image_dark, image_importance, image_depth]

    output_data = model.predict(input_data)[slice]
    # put channels into first dimension to force them as array
    output_data = np.swapaxes(output_data, 1, 2)
    output_data = np.swapaxes(output_data, 0, 1)

    #np.save(f'output_{modelName}_{sample_id}_s{slice_id}.npy', output_data)

    return output_data

slices = []
#model = keras.models.load_model(f'model_{modelName}.h5')
for slice in range(16):
    # load the model
    model = keras.models.load_model(f'model{slice}_{modelName}.h5')
    slices.append(process_sample(model, slice))

# combine the slices
output_data = np.concatenate(slices, axis=0)
np.save(f'output_{modelName}_{sample_id}.npy', output_data)