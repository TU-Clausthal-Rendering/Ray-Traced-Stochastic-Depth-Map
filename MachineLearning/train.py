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

tf.keras.utils.set_random_seed(2) # use same random seed for training

# returns true if sample was processed, false if sample was not processed because it does not exist
def process_sample(model : keras.models.Model, epochs):
    # check if f'{dataPath}bright_{sample_id}.npy' exists
    if not os.path.isfile(f'{dataPath}bright_.npy'):
        return False

    # Load and preprocess input images
    image_bright = np.load(f'{dataPath}bright_.npy')
    image_dark = np.load(f'{dataPath}dark_.npy')
    image_ref = np.load(f'{dataPath}ref_.npy')
    #image_depth = np.load(f'{dataPath}depth_.npy')
    image_invDepth = np.load(f'{dataPath}invDepth_.npy')

    # arrays have uint values 0 - 255. Convert to floats 0.0 - 1.0
    image_bright = image_bright.astype(np.float32) / 255.0
    image_dark = image_dark.astype(np.float32) / 255.0
    image_ref = image_ref.astype(np.float32) / 255.0
    image_importance = np.ones(image_bright.shape, dtype=np.float32) - (image_bright - image_dark) # importance = bright - dark
    #image_importance = np.zeros(image_bright.shape, dtype=np.float32) # importance = 0

    # Preprocess images and expand dimensions
    input_data = [image_bright, image_dark, image_importance, image_invDepth]
    #target_data = np.expand_dims(image_ref, axis=0)
    target_data = image_ref

    for epoch in range(epochs):
        model.fit(input_data, target_data, batch_size=128, initial_epoch=epoch, epochs=epoch+1)
        # save intermediate model
        model.save('model_autosave.h5')

    return True

def build_network():
    # determine size of convolutional network
    img_shape = np.load(f'{dataPath}bright_.npy').shape[1:]
    print("image shape: ", img_shape)

    unorm_relu = keras.layers.ReLU(max_value=None) # this relu cuts off below 0.0 and above 1.0

    # two inputs
    layer_input_bright = keras.layers.Input(shape=(img_shape[0], img_shape[1], 1))
    layer_input_dark = keras.layers.Input(shape=(img_shape[0], img_shape[1], 1))
    layer_input_importance = keras.layers.Input(shape=(img_shape[0], img_shape[1], 1))
    layer_input_depth = keras.layers.Input(shape=(img_shape[0], img_shape[1], 1))
    # concatenate inputs
    layer_concat = keras.layers.Concatenate(axis=-1)([layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth])
    # conv2d
    layer_conv2d_1 = keras.layers.Conv2D(16, kernel_size=(3, 3), activation=unorm_relu, padding='same')(layer_concat)
    layer_conv2d_2 = keras.layers.Conv2D(4, kernel_size=(3, 3), activation=unorm_relu, padding='same')(layer_conv2d_1)
    layer_conv2d_3 = keras.layers.Conv2D(1, kernel_size=3, activation='linear', padding='same')(layer_conv2d_2)
    # clamp layer between layer_input_dark and layer_input_bright
    layer_min = keras.layers.Minimum()([layer_conv2d_3, layer_input_bright])
    layer_minmax = keras.layers.Maximum()([layer_min, layer_input_dark])

    eval_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=layer_minmax # use clamping for evaluation
    )

    train_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=layer_conv2d_3 # use unclamped output for training
    )

    # Compile the model
    train_model.compile(optimizer='nadam', loss='mean_squared_error')

    return (train_model, eval_model)


train_model, eval_model = build_network()
process_sample(train_model, 10)

# save the model
eval_model.save('model.h5')