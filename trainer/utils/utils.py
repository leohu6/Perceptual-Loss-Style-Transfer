import tensorflow as tf

style_layers = ['block1_conv2',
                'block2_conv2',
                'block3_conv3',
                'block4_conv3']

content_layers = ['block2_conv2']

vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
vgg16.trainable = False

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2]*input_shape[3], tf.float32)
  return result/(num_locations)

def preprocess_vgg(inputs):
  r, g, b = tf.split(axis=3, num_or_size_splits=3, value=inputs)
  VGG_MEAN = [103.939, 116.779, 123.68]
  bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
  return bgr

def vgg_layers(layers):
  outputs = [vgg16.get_layer(name).output for name in layers]
  model = tf.keras.Model([vgg16.input], outputs=outputs)
  return model