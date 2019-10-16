import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from trainer import utils
from trainer import models

coco_dataset = tfds.load('coco/2017', split=tfds.Split.TRAIN, data_dir='gs://duke-tfds')

# load starry night

model = transformer_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=full_loss(starry_night))
model.fit(images, epochs=50, steps_per_epoch=None, use_multiprocessing=False)