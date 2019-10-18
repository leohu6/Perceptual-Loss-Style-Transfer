import tensorflow as tf
from .normalizations import InstanceNormalization
from .MyInstanceNorm import MyInstanceNorm

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    
    def get_config(self):
        config = {
            'padding':
            self.padding
        }
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def normalization(intput_tensor, method='instance'):
  if method == 'instance':
    x = MyInstanceNorm(center=True, scale=True,
                                                  beta_initializer="random_uniform",
                                                  gamma_initializer="random_uniform")(intput_tensor)
  else:
    x = tf.keras.layers.BatchNormalization()(intput_tensor)
  return x

def conv_w_reflection(input_tensor,
               kernel_size,
               filters,
               stride):
  p = kernel_size // 2
  x = ReflectionPadding2D(padding=(p, p))(input_tensor)
  x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
  x = normalization(x, method='instance')
  x = tf.keras.layers.Activation(tf.nn.relu)(x)
  return x

def conv_block(input_tensor, filters):
  x = ReflectionPadding2D(padding=(1, 1))(input_tensor)
  x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), use_bias=False)(x)
  x = normalization(x, method='instance')
  x = tf.keras.layers.Activation(tf.nn.relu)(x)

  x = ReflectionPadding2D(padding=(1, 1))(x)
  x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), use_bias=False)(x)
  x = normalization(x, method='instance')
  return x

def residual_block(input_tensor, filters):
  b1 = conv_block(input_tensor, filters)
  x = tf.keras.layers.Add()([input_tensor, b1])
  return x

def upsample_conv(input_tensor, kernel_size, filters, stride):
  x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=stride, padding='same', use_bias=False)(input_tensor)
  x = normalization(x, method='instance')
  x = tf.keras.layers.Activation(tf.nn.relu)(x)
  return x

def transformer_model():
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    print('created')
    x = conv_w_reflection(inputs, 9, 32, 1)
    x = conv_w_reflection(x, 3, 64, 2)
    x = conv_w_reflection(x, 3, 128, 2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = upsample_conv(x, 3, 64, 2)
    x = upsample_conv(x, 3, 32, 2)
    x = tf.keras.layers.Conv2DTranspose(3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.scalar_mul(255./2, x) + 255./2)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
