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

class WeightedSumLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(WeightedSumLayer, self).__init__(name=name)

    def call(self, inputs):
        v, w = inputs
        weighted_sum = tf.reduce_sum(v * w, axis=-1)
        sum_w = tf.reduce_sum(w, axis=-1)
        #output = weighted_sum
        output = weighted_sum / (sum_w + 1e-8)
        output = tf.expand_dims(output, axis=-1)
        return output
    

class AoLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = keras.losses.MeanSquaredError()

    @tf.function
    def call(self, y_true, y_pred):
        y_ref = y_true[:, :, :, :, 0]
        y_max_error = y_true[:, :, :, :, 1]

        # multiply with max error
        y_ref = tf.math.multiply(y_ref, y_max_error)
        y_pred = tf.math.multiply(y_pred, y_max_error)
        
        y_diff = tf.math.subtract(y_ref, y_pred)
        # multiply with 2.0 if y_diff is positive
        y_diff = tf.where(y_diff > 0, y_diff * 2, y_diff)
        
        return tf.math.square(y_diff) # return squared error
    
    def get_config(self):
        config = super().get_config()
        return config
