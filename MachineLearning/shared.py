from tensorflow import keras
import tensorflow as tf

class RelativeDepthLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(RelativeDepthLayer, self).__init__(name=name)
    
    def call(self, inputs):
        # assume inputs is a list of 2 tensors
        original = inputs[0]
        others = inputs[1]

        new_tensors = [None]*others.shape[3]
        # others is a tensor of shape (batch, height, width, channels)
        for i in range(len(new_tensors)):
            # relative_depth = d_other / d_original
            new_tensors[i] = tf.math.divide(others[:, :, :, i], original[:, :, :, 0])
            # 1 - relative_depth
            new_tensors[i] = tf.math.subtract(tf.ones_like(new_tensors[i]), new_tensors[i])
            # absolute value
            new_tensors[i] = tf.math.abs(new_tensors[i])
        return tf.stack(new_tensors, axis=3)

    def get_config(self):
        config = super(RelativeDepthLayer, self).get_config()
        return config