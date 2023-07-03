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
def process_sample(model : keras.models.Model, epochs, slice):
    # check if f'{dataPath}bright_{sample_id}.npy' exists
    if not os.path.isfile(f'{dataPath}bright_.npy'):
        return False

    # Load and preprocess input images
    image_bright = np.load(f'{dataPath}bright_s{slice}.npy')
    image_dark = np.load(f'{dataPath}dark_s{slice}.npy')
    image_ref = np.load(f'{dataPath}ref_s{slice}.npy')
    image_depth = np.load(f'{dataPath}depth_s{slice}.npy')
    
    image_bright = image_bright.astype(np.float32) / 255.0
    image_dark = image_dark.astype(np.float32) / 255.0
    image_ref = image_ref.astype(np.float32) / 255.0
    image_max_error = np.maximum(image_bright - image_dark, 0.05)
    image_importance = np.ones(image_bright.shape, dtype=np.float32) - image_max_error # importance = bright - dark

    # Preprocess images and expand dimensions
    input_data = [image_bright, image_dark, image_importance, image_depth]
    #target_data = np.expand_dims(image_ref, axis=0)
    #target_data = image_ref
    #target_data = [image_ref, image_max_error]
    target_data = tf.stack([image_ref, np.square(image_max_error)], axis=4)

    models['train'].fit(input_data, target_data, batch_size=16, epochs=epochs)
    for name, model in models.items():
        model.save(f'model{slice}_{name}.h5')

    return True


def distance_kernel(offset):
    return math.exp(-0.5 * offset * offset / 26.0)

def create_conv2d_single_channel_extractor(kernel_size, horizontal : bool):
    kernel = (kernel_size, 1)    
    if horizontal:
        kernel = (1, kernel_size)

    conv = keras.layers.Conv2D(kernel_size, kernel_size=kernel, activation='linear', padding='same', use_bias=False)

    return conv

def set_conv2d_single_channel_extractor_weights(conv, kernel_size, horizontal : bool, trainable : bool):
    kernel = (kernel_size, 1)    
    if horizontal:
        kernel = (1, kernel_size)

    weights = np.zeros((kernel[0], kernel[1], 1, kernel_size))
    if horizontal:
        for kx in range(kernel_size):
            weights[0, kx, 0, kx] = 1.0
    else:
        for ky in range(kernel_size):
            weights[ky, 0, 0, ky] = 1.0
    
    conv.set_weights([weights])
    conv.trainable = trainable

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

    kernel_size = 9


    # split depth into 9 channels for 3x3 convolution
    pick_depth_conv_x = create_conv2d_single_channel_extractor(kernel_size, True)
    pick_depth_layer_x = pick_depth_conv_x(layer_input_depth)
    relative_depth_layer_x = RelativeDepthLayer()([layer_input_depth, pick_depth_layer_x])

    pick_depth_conv_y = create_conv2d_single_channel_extractor(kernel_size, False)
    pick_depth_layer_y = pick_depth_conv_y(layer_input_depth)
    relative_depth_layer_y = RelativeDepthLayer()([layer_input_depth, pick_depth_layer_y])

    # split importance into 9 channels for 3x3 convolution
    pick_importance_conv_x = create_conv2d_single_channel_extractor(kernel_size, True)
    pick_importance_layer_x = pick_importance_conv_x(layer_input_importance)

    pick_importance_conv_y = create_conv2d_single_channel_extractor(kernel_size, False)
    pick_importance_layer_y = pick_importance_conv_y(layer_input_importance)

    pick_bright_conv_x = create_conv2d_single_channel_extractor(kernel_size, True)
    #pick_bright_layer_x = pick_bright_conv_x(layer_input_bright)
    pick_bright_layer_x = pick_bright_conv_x(tf.keras.layers.Average()([layer_input_bright, layer_input_dark]))

    train_weights_conv_x = keras.layers.Conv2D(kernel_size, kernel_size=1, activation='sigmoid', padding='same', kernel_initializer='zeros', bias_initializer='zeros')
    train_weights_layer_x = train_weights_conv_x(keras.layers.Concatenate(axis=-1)([pick_depth_layer_x, pick_importance_layer_x]))

    train_weights_conv_y = keras.layers.Conv2D(kernel_size, kernel_size=1, activation='sigmoid', padding='same', kernel_initializer='zeros', bias_initializer='zeros')
    train_weights_layer_y = train_weights_conv_y(keras.layers.Concatenate(axis=-1)([pick_depth_layer_y, pick_importance_layer_y]))

    weights_x_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=train_weights_layer_x
    )

    weights_y_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=train_weights_layer_y
    )

    weighted_sum_x = WeightedSumLayer()([pick_bright_layer_x, train_weights_layer_x])

    pick_bright_conv_y = create_conv2d_single_channel_extractor(kernel_size, False)
    pick_bright_layer_y = pick_bright_conv_y(weighted_sum_x)

    weighted_sum_y = WeightedSumLayer()([pick_bright_layer_y, train_weights_layer_y])

    weighted_x_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=weighted_sum_x
    )

    weighted_y_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=weighted_sum_y
    )

    # finally clamp between bright an dark
    layer_min = keras.layers.Minimum()([weighted_sum_y, layer_input_bright])
    layer_minmax = keras.layers.Maximum()([layer_min, layer_input_dark])

    train_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=weighted_sum_y
    )

    eval_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=layer_minmax
    )

    models = {
        'weights_x': weights_x_model,
        'weights_y': weights_y_model,
        'weighted_x': weighted_x_model,
        'weighted_y': weighted_y_model,
        'eval': eval_model,
    }

    #w = pick_depth_conv.get_weights()
    # initialize weights
    set_conv2d_single_channel_extractor_weights(pick_depth_conv_x, kernel_size, True, False)
    set_conv2d_single_channel_extractor_weights(pick_importance_conv_x, kernel_size, True, False)
    set_conv2d_single_channel_extractor_weights(pick_bright_conv_x, kernel_size, True, False)
    set_conv2d_single_channel_extractor_weights(pick_depth_conv_y, kernel_size, False, False)
    set_conv2d_single_channel_extractor_weights(pick_importance_conv_y, kernel_size, False, False)
    set_conv2d_single_channel_extractor_weights(pick_bright_conv_y, kernel_size, False, False)

    for model in models.values():
        model.compile(optimizer='adam', loss='mean_squared_error')

    # special loss for train model
    loss = AoLoss()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    train_model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)
    models['train'] = train_model

    return models


models = build_network()
process_sample(models, 32, 0)