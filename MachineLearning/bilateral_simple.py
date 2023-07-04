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

class PrintWeightsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        print(" depth variance: ", weights[0][0], " importance variance: ", weights[1][0])

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
    
    image_bright = image_bright.astype(np.float32) / 255.0
    image_dark = image_dark.astype(np.float32) / 255.0
    image_ref = image_ref.astype(np.float32) / 255.0
    image_max_error = np.maximum(image_bright - image_dark, 0.001)
    #image_importance = np.ones(image_bright.shape, dtype=np.float32) - image_max_error # importance = bright - dark

    # Preprocess images and expand dimensions
    input_data = [image_bright, image_dark, image_max_error, image_depth]
    #target_data = np.expand_dims(image_ref, axis=0)
    #target_data = image_ref
    #target_data = [image_ref, image_max_error]
    target_data = tf.stack([image_ref, np.square(image_max_error)], axis=4)

    models['train'].fit(input_data, target_data, batch_size=16, epochs=epochs, callbacks=[PrintWeightsCallback()])
    for name, model in models.items():
        model.save(f'model_{name}.h5')

    #print("final weights (depth, importance): ", models['train'].get_weights())

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

    kernel_radius = 4
    kernel_size = kernel_radius * 2 + 1
    


    # split depth into 9 channels for 3x3 convolution
    pick_depth_conv_x = NeighborExpansionLayer(kernel_radius, 2) 
    pick_depth_layer_x = pick_depth_conv_x(layer_input_depth)
    relative_depth_layer_x = RelativeDepthLayer()([layer_input_depth, pick_depth_layer_x])

    pick_depth_conv_y = NeighborExpansionLayer(kernel_radius, 1)
    pick_depth_layer_y = pick_depth_conv_y(layer_input_depth)
    relative_depth_layer_y = RelativeDepthLayer()([layer_input_depth, pick_depth_layer_y])

    # split importance into 9 channels for 3x3 convolution
    pick_importance_conv_x = NeighborExpansionLayer(kernel_radius, 2)
    pick_importance_layer_x = pick_importance_conv_x(layer_input_importance)

    pick_importance_conv_y = NeighborExpansionLayer(kernel_radius, 1) 
    pick_importance_layer_y = pick_importance_conv_y(layer_input_importance)

    pick_bright_conv_x = NeighborExpansionLayer(kernel_radius, 2) 
    pick_bright_layer_x = pick_bright_conv_x(tf.keras.layers.Average()([layer_input_bright, layer_input_dark]))

    depth_gaussion_conv = GaussianActivation(variance_initializer=keras.initializers.Constant(0.04))
    depth_gaussian_layer_x = depth_gaussion_conv(relative_depth_layer_x)
    depth_gaussian_layer_y = depth_gaussion_conv(relative_depth_layer_y)

    importance_gaussion_conv = GaussianActivation(variance_initializer=keras.initializers.Constant(0.028))
    importance_gaussian_layer_x = importance_gaussion_conv(pick_importance_layer_x)
    importance_gaussian_layer_y = importance_gaussion_conv(pick_importance_layer_y)

    # multiply depth and importance
    multiplied_weights_layer_x = keras.layers.Multiply()([depth_gaussian_layer_x, importance_gaussian_layer_x])
    weighted_sum_layer_x = WeightedSumLayer()([pick_bright_layer_x, multiplied_weights_layer_x])

    pick_bright_conv_y = NeighborExpansionLayer(kernel_radius, 1)
    pick_bright_layer_y = pick_bright_conv_y(weighted_sum_layer_x)

    multiplied_weights_layer_y = keras.layers.Multiply()([depth_gaussian_layer_y, importance_gaussian_layer_y])
    weighted_sum_y = WeightedSumLayer()([pick_bright_layer_y, multiplied_weights_layer_y])

    weighted_x_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=multiplied_weights_layer_x
    )

    weighted_y_model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=multiplied_weights_layer_y
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
        'weighted_x': weighted_x_model,
        'weighted_y': weighted_y_model,
        'eval': eval_model,
    }

    for model in models.values():
        # add a callback function that prints the weights after each epoch
        model.compile(optimizer='adam', loss='mean_squared_error')

    # special loss for train model
    loss = AoLoss()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    train_model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)
    models['train'] = train_model

    return models


models = build_network()
process_sample(models, 128)