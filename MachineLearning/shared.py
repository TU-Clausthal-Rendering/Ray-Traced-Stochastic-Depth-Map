from tensorflow import keras
import tensorflow as tf
import numpy as np

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
        y_max_error = tf.square(y_max_error) # square max error

        # multiply with max error
        y_ref = tf.math.multiply(y_ref, y_max_error)
        y_pred = tf.math.multiply(y_pred, y_max_error)
        
        y_diff = tf.math.subtract(y_ref, y_pred)
        # multiply with 2.0 if y_diff is positive
        #y_diff = tf.where(y_diff > 0, y_diff * 2, y_diff)
        
        # errors below 0.01 are irrelevant
        y_diff = tf.maximum(tf.abs(y_diff) - 0.01, 0.0)

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

class GreaterThanConstraint(tf.keras.constraints.Constraint):
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def __call__(self, w):
        return tf.keras.backend.clip(w, self.epsilon, None)

class BilateralBlur(tf.keras.layers.Layer):
    def __init__(self, R=2, **kwargs):
        super(BilateralBlur, self).__init__(**kwargs)
        self.R = R

    def build(self, input_shape):
        self.kernel_size = 2 * self.R + 1
        self.depth_variance = self.add_weight(
            name='depth_variance', 
            #initializer=keras.initializers.Constant(0.001),
            initializer=keras.initializers.Constant(0.0004),
            constraint=GreaterThanConstraint(epsilon=1e-7),
            trainable=True
        )
        self.spatial_variance = self.add_weight(
            name='spatial_variance', 
            #initializer=keras.initializers.Constant(10.0),
            initializer=keras.initializers.Constant(13.16),
            constraint=GreaterThanConstraint(epsilon=1e-7),
            trainable=True
        )
        #self.importance_exponent = self.add_weight(
        #    name='importance_exponent',
        #    initializer=keras.initializers.Constant(2.37),
        #    constraint=GreaterThanConstraint(epsilon=1e-7),
        #)
        self.dev_exponent = self.add_weight(
            name='dev_exponent',
            #initializer=keras.initializers.Constant(2.0),
            initializer=keras.initializers.Constant(0.3),
            constraint=GreaterThanConstraint(epsilon=1e-7) 
        )
        self.dark_epsilon = self.add_weight(
            name='dark_epsilon',
            initializer=keras.initializers.Constant(0.375),
            constraint=GreaterThanConstraint(epsilon=1e-8),
        )
        self.contrast_enhance = self.add_weight(
            name='contrast_enhance',
            initializer=keras.initializers.Constant(0.95),
            constraint=GreaterThanConstraint(epsilon=0.1),
        )

        # spatial distances (-2, -1, 0, 1, 2)
        self.spatial_dist = tf.constant([x - self.R for x in range(self.kernel_size)], 
                                        shape=(1,1,1,self.kernel_size),
                                        dtype=tf.float32)
        

    def custom_pow(self, a, b):
        return tf.math.exp(tf.math.multiply(b, tf.math.log(tf.maximum(a, 1e-8))))

    def do_blur(self, bright_x, dark_x, depths_x, depths):
        # convert depths_x to relative depths
        rel_depth_x = tf.minimum(tf.abs(tf.divide(depths_x, depths) - 1.0), 1.0)
        #rel_depth_x = tf.abs(tf.divide(depths_x, depths) - 1.0)

        # apply a gaussian kernel to rel_depth_x    
        w_depth = tf.exp(-tf.square(rel_depth_x) / (2 * self.depth_variance))
        # apply gaussian kernel to spatial distances
        w_spatial = tf.exp(-tf.square(self.spatial_dist) / (2 * self.spatial_variance))
        # calc importance weights
        #w_importance = tf.minimum(1.0 - (bright_x - dark_x), 1.0)
        #w_importance = self.custom_pow(w_importance, self.importance_exponent)

        # apply spatial weights
        w_x = w_depth * w_spatial #* w_importance

        # normalize the weights
        w_x = tf.divide(w_x, tf.reduce_sum(w_x, axis=-1, keepdims=True))

        # apply weights and reduce to channel size 1 (dot product)
        bright_x = tf.reduce_sum(tf.multiply(bright_x, w_x), axis=-1, keepdims=True)
        dark_x = tf.reduce_sum(tf.multiply(dark_x, w_x), axis=-1, keepdims=True)

        return bright_x, dark_x



    def call(self, inputs):
        bright, dark, depths = inputs # color = AO bright/dark values

        # prepare for blur in X
        bright_x = tf.image.extract_patches(bright, sizes=[1, 1, self.kernel_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
        dark_x = tf.image.extract_patches(dark, sizes=[1, 1, self.kernel_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
        depths_x = tf.image.extract_patches(depths, sizes=[1, 1, self.kernel_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')

        bright_x, dark_x = self.do_blur(bright_x, dark_x, depths_x, depths)

        # do the same for Y direction
        bright_y = tf.image.extract_patches(bright_x, sizes=[1, self.kernel_size, 1, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
        dark_y = tf.image.extract_patches(dark_x, sizes=[1, self.kernel_size, 1, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
        depths_y = tf.image.extract_patches(depths, sizes=[1, self.kernel_size, 1, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')

        bright_mean, dark_mean = self.do_blur(bright_y, dark_y, depths_y, depths)

        # compute the local deviation
        dev_bright = self.custom_pow(tf.abs(bright - bright_mean), self.dev_exponent)
        dev_dark = self.custom_pow(tf.abs(dark - dark_mean), self.dev_exponent)
        # prevent division by zero
        dev_dark = tf.maximum(dev_dark, self.dark_epsilon)
        # enhance dark contrast
        dev_bright = dev_bright * self.contrast_enhance
        # normalize deviations
        weight_sum = tf.add(dev_bright, dev_dark)
        dev_bright = tf.divide(dev_bright, weight_sum)
        dev_dark = tf.divide(dev_dark, weight_sum)

        res = dev_dark * bright + dev_bright * dark
        return res

def get_custom_objects():
    return {
        'RelativeDepthLayer': RelativeDepthLayer,
        'WeightedSumLayer': WeightedSumLayer,
        'AoLoss': AoLoss,
        'GaussianActivation': GaussianActivation,
        'NeighborExpansionLayer': NeighborExpansionLayer,
        'GreaterThanConstraint': GreaterThanConstraint,
        'BilateralBlur': BilateralBlur
    }