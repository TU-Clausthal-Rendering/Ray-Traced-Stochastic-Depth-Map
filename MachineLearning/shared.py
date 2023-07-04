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



class GaussianActivation(keras.layers.Layer):
    def __init__(self, variance_initializer='ones', **kwargs):
        super(GaussianActivation, self).__init__(**kwargs)
        self.variance_initializer = tf.keras.initializers.get(variance_initializer)

    def build(self, input_shape):
        self.variance = self.add_weight(
            shape=(1,),
            initializer=self.variance_initializer,
            name='variance',
            trainable=True
        )
        super(GaussianActivation, self).build(input_shape)

    def call(self, inputs):
        exponent = -tf.square(inputs) / (2 * self.variance)
        activation = tf.exp(exponent)
        return activation

    def get_config(self):
        config = super(GaussianActivation, self).get_config()
        config.update({
            'variance_initializer': tf.keras.initializers.serialize(self.variance_initializer),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        variance_initializer = tf.keras.initializers.deserialize(config.pop('variance_initializer'))
        instance = cls(variance_initializer=variance_initializer, **config)
        return instance

class NeighborExpansionLayer(keras.layers.Layer):
    def __init__(self, radius, dimension, **kwargs):
        super(NeighborExpansionLayer, self).__init__(**kwargs)
        self.radius = radius
        self.dimension = dimension

    @tf.function
    def call(self, inputs):
        # pad the dimension that will be expanded/shifted
        paddings = None
        if self.dimension == 1:
            paddings = tf.constant([[0, 0], [self.radius, self.radius], [0, 0], [0, 0]])
        elif self.dimension == 2:
            paddings = tf.constant([[0, 0], [0, 0], [self.radius, self.radius], [0, 0]])
        else:
            raise ValueError("Invalid dimension. Use 1 for height or 2 for width.")

        padded_inputs = tf.pad(inputs, paddings, mode='CONSTANT') # CONSTANT = pad with 0s

        expanded_channels = []
        for r in range(-self.radius, self.radius + 1):
            shifted_inputs = tf.roll(padded_inputs, shift=-r, axis=self.dimension)
            if self.dimension == 1:
                expanded_channels.append(shifted_inputs[:, self.radius:-self.radius, :, :])
            elif self.dimension == 2:
                expanded_channels.append(shifted_inputs[:, :, self.radius:-self.radius, :])

        expanded_tensor = tf.concat(expanded_channels, axis=-1)
        return expanded_tensor


def get_custom_objects():
    return {
        'RelativeDepthLayer': RelativeDepthLayer,
        'WeightedSumLayer': WeightedSumLayer,
        'AoLoss': AoLoss,
        'GaussianActivation': GaussianActivation,
        'NeighborExpansionLayer': NeighborExpansionLayer
    }