import tensorflow as tf
import matplotlib.pyplot as plt

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

def load_img(path_to_img):
  max_dim = 2048
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  img = img[tf.newaxis, :]
  img = tf.cast(img, tf.float32)
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  plt.imshow(image)
  if title:
    plt.title(title)

def scan_convert(image, irad, frad, iang, fang):
  """Scan converts beam lines"""
  image, _ = polarTransform.convertToCartesianImage(
      np.transpose(image),
      initialRadius=irad,
      finalRadius=frad,
      initialAngle=iang,
      finalAngle=fang,
      hasColor=False,
      order=1)
  return np.transpose(image[:, int(irad):])
  
def process(ele):
  """Cuts to -80 dB and normalizes images from 0 to 1"""
  ele['das'] = tf.reshape(ele['das']['dB'], [ele['height'], ele['width']])
  ele['das'] = tf.clip_by_value(ele['das'], -80, 0)
  ele['das'] = (ele['das'] - tf.reduce_min(ele['das']))/(tf.reduce_max(ele['das']) - tf.reduce_min(ele['das']))

  ele['dtce'] = tf.reshape(ele['dtce'], [ele['height'], ele['width']])
  ele['dtce'] = (ele['dtce'] - tf.reduce_min(ele['dtce']))/(tf.reduce_max(ele['dtce']) - tf.reduce_min(ele['dtce']))
  image = tf.image.resize(ele['das'][..., None], [512, 512])
  image = tf.image.grayscale_to_rgb(image)
  print(image.shape)
  return image, image

def just_das(ele):
  converted = scan_convert(ele['das'].numpy(),
                           ele['initial_radius'].numpy(),
                           ele['final_radius'].numpy(),
                           ele['initial_angle'].numpy(),
                           ele['final_angle'].numpy())
  return converted, converted