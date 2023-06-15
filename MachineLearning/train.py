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
def process_sample(sample_id, model : keras.models.Model, epoch):
    # check if f'{dataPath}bright_{sample_id}.npy' exists
    if not os.path.isfile(f'{dataPath}bright_{sample_id}.npy'):
        return False

    # Load and preprocess input images
    image_bright = np.load(f'{dataPath}bright_{sample_id}.npy')
    image_dark = np.load(f'{dataPath}dark_{sample_id}.npy')
    image_ref = np.load(f'{dataPath}ref_{sample_id}.npy')
    #image_depth = np.load(f'{dataPath}depth_{sample_id}.npy')
    image_invDepth = np.load(f'{dataPath}invDepth_{sample_id}.npy')

    # arrays have uint values 0 - 255. Convert to floats 0.0 - 1.0
    image_bright = image_bright.astype(np.float32) / 255.0
    image_dark = image_dark.astype(np.float32) / 255.0
    image_ref = image_ref.astype(np.float32) / 255.0
    image_importance = np.ones(image_bright.shape, dtype=np.float32) - (image_bright - image_dark) # importance = bright - dark
    #image_importance = np.zeros(image_bright.shape, dtype=np.float32) # importance = 0

    # Preprocess images and expand dimensions
    #input_data = np.expand_dims(np.concatenate((image_bright, image_dark), axis=2), axis=0)
    #input_data = [np.expand_dims(image_bright, axis=0), np.expand_dims(image_dark, axis=0), np.expand_dims(image_importance, axis=0), np.expand_dims(image_invDepth, axis=0)]
    input_data = [image_bright, image_dark, image_importance, image_invDepth]
    #target_data = np.expand_dims(image_ref, axis=0)
    target_data = image_ref

    model.fit(input_data, target_data, batch_size=16, epochs=1)
    return True

def build_network():
    # determine size of convolutional network
    img_shape = np.load(f'{dataPath}bright_{0}.npy').shape[1:]
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

    model = keras.models.Model(
        inputs=[layer_input_bright, layer_input_dark, layer_input_importance, layer_input_depth],
        outputs=layer_minmax
        #outputs=layer_conv2d_3
    )

    # Compile the model
    model.compile(optimizer='nadam', loss='mean_squared_error')
    return model


model = build_network()
#model = keras.models.load_model('model.h5')
for epoch in range(0, 100):
    print(f'epoch {epoch}')
    sample_id = 0
    while process_sample(sample_id, model, epoch):
        sample_id += 1

# save the model
model.save('model.h5')

# Generate output prediction
#output_data = model.predict(input_data)
#
#png_data = output_data[0]
#
## clip png data between image_dark and image_bright
##png_data = np.clip(png_data, image_dark, image_bright)
#
## remove/flatten the last dimension
#png_data = png_data.reshape((img_shape[0], img_shape[1]))
#
## convert to uint 0 - 255
#png_data = (png_data * 255).astype(np.uint8)
#np.save(f'test.npy', png_data)

# output model weights
weights = model.get_weights()
for i in range(len(weights)):
    type = "weight"
    if i % 2 == 1:
        type = "bias"
    
    # save as numpy array
    np.save(f'{type}_{i//2}.npy', weights[i])