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

# load the model
model = keras.models.load_model('model_autosave.h5')

# returns true if sample was processed, false if sample was not processed because it does not exist
def process_sample(sample_id, model : keras.models.Model):
    # check if f'{dataPath}bright_{sample_id}.npy' exists
    if not os.path.isfile(f'{dataPath}bright_{sample_id}.npy'):
        return False

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

    #model.fit(input_data, target_data, batch_size=16, epochs=1)
    output_data = model.predict(input_data)

    np.save(f'output_{sample_id}.npy', output_data)

    return True

#sample_id = 0
#process_sample(sample_id, model)


# output model weights
weights = model.get_weights()
for i in range(len(weights)):
    type = "weight"
    if i % 2 == 1:
        type = "bias"
    
    # save as numpy array
    np.save(f'{type}_{i//2}.npy', weights[i])