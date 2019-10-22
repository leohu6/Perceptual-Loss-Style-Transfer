import tensorflow as tf
from .utils import preprocess_vgg
from .utils import gram_matrix
from .utils import vgg_layers

style_layers = ['block1_conv2',
                'block2_conv2',
                'block3_conv3',
                'block4_conv3']

content_layers = ['block2_conv2']

vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
vgg16.trainable = False

def style_loss(style_img):
    def loss(y_true, y_pred):
      vgg_out = vgg_layers(style_layers)
      style_img_processed = preprocess_vgg(style_img)
      style_out = vgg_out(style_img_processed)
      style_outputs = [gram_matrix(style) for style in style_out]
        
      y_pred_processed = preprocess_vgg(y_pred)
      predicted_out = vgg_out(y_pred_processed)
      predicted_outputs = [gram_matrix(predicted) for predicted in predicted_out]
    
      cum_loss = 0
      for predicted_gram, style_gram in zip(predicted_outputs, style_outputs):
        gram_shape = tf.shape(style_gram)
        new_l = tf.keras.losses.MeanSquaredError()(predicted_gram, style_gram)
        cum_loss += new_l
      return cum_loss
    return loss

def content_loss_test(content_img):
    def loss(y_true,y_pred):
      vgg_out = vgg_layers(content_layers)
      content_img_processed = preprocess_vgg(content_img)
      content_out = vgg_out(content_img_processed)
      
      y_pred_processed = preprocess_vgg(y_pred)
      predicted_out = vgg_out(y_pred_processed)
      input_shape = tf.shape(predicted_out)

      new_l = tf.keras.losses.MeanSquaredError()(predicted_out, content_out)
      return new_l
    return loss

def content_loss(y_true, y_pred):
    vgg_out = vgg_layers(content_layers)
    content_img_processed = preprocess_vgg(y_true)
    content_out = vgg_out(content_img_processed)
      
    y_pred_processed = preprocess_vgg(y_pred)
    predicted_out = vgg_out(y_pred_processed)

    new_l = tf.keras.losses.MeanSquaredError()(predicted_out, content_out)
    return new_l
    
def tv_loss(y_true, y_pred):
    return tf.image.total_variation(y_pred)

def full_loss(style_img):
    def loss(y_true,y_pred):
      STYLE_WEIGHT = 5
      CONTENT_WEIGHT = 1e0
      TV_WEIGHT = 1e-4
    
      style_l = style_loss(style_img)(y_true, y_pred)
      content_l = content_loss_test(y_true)(y_true, y_pred)
      tv_loss = tf.image.total_variation(y_pred)
    
      return STYLE_WEIGHT * style_l + CONTENT_WEIGHT * content_l + TV_WEIGHT * tv_loss
    return loss