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

sample_id = 0
slice_id = 0

# set current directory as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tf.keras.utils.set_random_seed(2) # use same random seed for training

# Load and preprocess input images
image_bright = np.array(Image.open(f'{dataPath}bright_{sample_id}_s{slice_id}.png').convert('L'))
image_dark = np.array(Image.open(f'{dataPath}dark_{sample_id}_s{slice_id}.png').convert('L'))
image_ref = np.array(Image.open(f'{dataPath}ref_{sample_id}_s{slice_id}.png').convert('L'))

img_shape = image_bright.shape
print("image shape: ", img_shape)

# arrays have uint values 0 - 255. Convert to floats 0.0 - 1.0
image_bright = image_bright.astype(np.float32) / 255.0
image_dark = image_dark.astype(np.float32) / 255.0
image_ref = image_ref.astype(np.float32) / 255.0

# create an extra dimension for the channel
image_bright = np.expand_dims(image_bright, axis=2)
image_dark = np.expand_dims(image_dark, axis=2)
image_ref = np.expand_dims(image_ref, axis=2)

# Preprocess images and expand dimensions
#input_data = np.expand_dims(np.concatenate((image_bright, image_dark), axis=2), axis=0)
input_data = [np.expand_dims(image_bright, axis=0), np.expand_dims(image_dark, axis=0)]
target_data = np.expand_dims(image_ref, axis=0)

# Define the CNN model architecture
#model = Sequential()
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_shape[0], img_shape[1], 2)))
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same'))

# two inputs
layer_input_bright = keras.layers.Input(shape=(img_shape[0], img_shape[1], 1))
layer_input_dark = keras.layers.Input(shape=(img_shape[0], img_shape[1], 1))
# concatenate inputs
layer_concat = keras.layers.Concatenate(axis=-1)([layer_input_bright, layer_input_dark])
# conv2d
layer_conv2d_1 = keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(layer_concat)
#layer_conv2d_2 = keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same')(layer_conv2d_1)
layer_conv2d_3 = keras.layers.Conv2D(1, kernel_size=3, activation='relu', padding='same')(layer_conv2d_1)
# clamp layer between layer_input_dark and layer_input_bright
layer_min = keras.layers.Minimum()([layer_conv2d_3, layer_input_bright])
layer_minmax = keras.layers.Maximum()([layer_min, layer_input_dark])

model = keras.models.Model(
    inputs=[layer_input_bright, layer_input_dark],
    outputs=layer_minmax
    #outputs=layer_conv2d_3
)

# Compile the model
model.compile(optimizer='nadam', loss='mean_squared_error')

# Train the model
model.fit(input_data, target_data, epochs=1000, batch_size=1)

# Generate output prediction
output_data = model.predict(input_data)

png_data = output_data[0]

# clip png data between image_dark and image_bright
#png_data = np.clip(png_data, image_dark, image_bright)

# remove/flatten the last dimension
png_data = png_data.reshape((img_shape[0], img_shape[1]))

# convert to uint 0 - 255
png_data = (png_data * 255).astype(np.uint8)

# Rescale and save the output image
output_image = Image.fromarray(png_data)
output_image.save('test.png')

# output model weights
weights = model.get_weights()
for i in range(len(weights)):
    type = "weight"
    if i % 2 == 1:
        type = "bias"
    
    # save as numpy array
    np.save(f'{type}_{i//2}.npy', weights[i])