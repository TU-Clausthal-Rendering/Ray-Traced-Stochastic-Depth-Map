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
from shared import *
import math

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
    image_depth = np.load(f'{dataPath}depth_.npy')
    #image_invDepth = np.load(f'{dataPath}invDepth_.npy')

    # arrays have uint values 0 - 255. Convert to floats 0.0 - 1.0
    image_bright = image_bright.astype(np.float32) / 255.0
    image_dark = image_dark.astype(np.float32) / 255.0
    image_ref = image_ref.astype(np.float32) / 255.0
    image_importance = np.ones(image_bright.shape, dtype=np.float32) - (image_bright - image_dark) # importance = bright - dark
    #image_importance = np.zeros(image_bright.shape, dtype=np.float32) # importance = 0

    # Preprocess images and expand dimensions
    input_data = [image_bright, image_dark, image_importance, image_depth]
    #target_data = np.expand_dims(image_ref, axis=0)
    target_data = image_ref

    for epoch in range(epochs):
        model.fit(input_data, target_data, batch_size=128, initial_epoch=epoch, epochs=epoch+1)
        # save intermediate model
        model.save('model_autosave.h5')

    return True


def distance_kernel(offset):
    return math.exp(-0.5 * offset * offset / 26.0)

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

    kernel_size = 5


    # split depth into 9 channels for 3x3 convolution
    pick_depth_conv = keras.layers.Conv2D(kernel_size*kernel_size, kernel_size=kernel_size, activation='linear', padding='same')
    pick_depth_layer = pick_depth_conv(layer_input_depth)

    relative_depth_layer = RelativeDepthLayer()([layer_input_depth, pick_depth_layer])

    # split importance into 9 channels for 3x3 convolution
    pick_importance_conv = keras.layers.Conv2D(kernel_size*kernel_size, kernel_size=kernel_size, activation='linear', padding='same')
    pick_importance_layer = pick_importance_conv(layer_input_importance)

    # split (pixel) distance weights into 9 channels for 3x3 convolution
    pick_distance_conv = keras.layers.Conv2D(kernel_size*kernel_size, kernel_size=kernel_size, activation='linear', padding='same')
    pick_distance_layer = pick_distance_conv(layer_input_depth) # input is irrelavnt, it will be overwritten by the bias

    depth_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=relative_depth_layer
    )

    importance_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=pick_importance_layer
    )

    distance_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=pick_distance_layer
    )

    models = {
        'depth': depth_model,
        'importance': importance_model,
        'distance': distance_model
    }

    #w = pick_depth_conv.get_weights()
    # initialize weights
    pick_weights = [np.zeros((kernel_size, kernel_size, 1, kernel_size * kernel_size)), np.zeros(kernel_size * kernel_size)]
    distance_weights = [np.zeros((kernel_size, kernel_size, 1, kernel_size * kernel_size)), np.zeros(kernel_size * kernel_size)]
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            pick_weights[0][ky, kx, 0, ky * kernel_size + kx] = 1.0
            distance_weights[1][ky * kernel_size + kx] = distance_kernel(kx - (kernel_size // 2)) * distance_kernel(ky - (kernel_size // 2))

    pick_depth_conv.set_weights(pick_weights)
    pick_depth_conv.trainable = False
    pick_importance_conv.set_weights(pick_weights)
    pick_importance_conv.trainable = False
    pick_distance_conv.set_weights(distance_weights)
    pick_distance_conv.trainable = False

    for model in models.values():
        model.compile(optimizer='adam', loss='mean_squared_error')

    return models


models = build_network()
#process_sample(train_model, 100)

# save the model
for name, model in models.items():
    model.save(f'model_{name}.h5')