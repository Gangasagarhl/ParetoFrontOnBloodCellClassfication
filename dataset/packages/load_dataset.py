import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


"""class load_data: 

    def __init__(self, train_data_path, validation_data_path, batch_size=32, image_height=224, image_width=224, seed=123):
        self.TRAIN_ROOT = train_data_path
        self.VALIDATION_ROOT = validation_data_path
        self.batch_size = batch_size
        self.img_h = image_height
        self.img_w = image_width
        self.seed = seed

    def to_grayscale(self, image, label):
        image = tf.image.rgb_to_grayscale(image)
        return image, label

    def load_data(self):
        train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
            self.TRAIN_ROOT,
            seed=self.seed,
            image_size=(self.img_h, self.img_w),
            batch_size=self.batch_size,
        )
        val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
            self.VALIDATION_ROOT,
            seed=self.seed,
            image_size=(self.img_h, self.img_w),
            batch_size=self.batch_size,
        )

        # Get class names BEFORE mapping to grayscale
        class_names = train_ds_raw.class_names

        # Convert datasets to grayscale
        train_ds = train_ds_raw.map(self.to_grayscale)
        val_ds = val_ds_raw.map(self.to_grayscale)

        return train_ds, val_ds, class_names
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class load_data: 

    def __init__(self, train_data_path, validation_data_path, batch_size=32, image_height=224, image_width=224, seed=123):
        self.TRAIN_ROOT = train_data_path
        self.VALIDATION_ROOT = validation_data_path
        self.batch_size = batch_size
        self.img_h = image_height
        self.img_w = image_width
        self.seed = seed

    def load_data(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.TRAIN_ROOT,
            seed=self.seed,
            image_size=(self.img_h, self.img_w),
            batch_size=self.batch_size,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.VALIDATION_ROOT,
            seed=self.seed,
            image_size=(self.img_h, self.img_w),
            batch_size=self.batch_size,
        )

        # Get class names (still in RGB)
        class_names = train_ds.class_names

        return train_ds, val_ds, class_names
