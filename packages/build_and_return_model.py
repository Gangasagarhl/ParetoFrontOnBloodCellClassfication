import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense, MaxPooling2D, GlobalAveragePooling2D
)


class build_model:
    def __init__(self):
        pass

    def build_model(self,
        dense_layers: int,
        list_number_of_nuerons_in_each_layer: list[int],
        Number_of_convoluational_filters: list[int],
        kernel_size: list[int]
    ) -> Sequential:
        """
        Builds a Sequential CNN + Dense model.
        
        Args:
        dense_layers: number of Dense layers (also equals number of classes/output units).
        list_number_of_nuerons_in_each_layer: neurons in each Dense layer, length == dense_layers.
        Number_of_convoluational_filters: list of filters for each Conv2D block.
        kernel_size: list of kernel sizes (ints) for each Conv2D block, same length as filters list.
        
        Returns:
        A compiled tf.keras.Sequential model.
        """
        model = Sequential()

        # — Convolutional blocks —
        for idx, (filters, k) in enumerate(zip(Number_of_convoluational_filters, kernel_size)):
            if idx == 0:
                # first conv needs input_shape
                model.add(Conv2D(filters, (k, k),
                                activation='relu',
                                input_shape=(224,224,1)))
                model.add(MaxPooling2D(pool_size=(2,2)))
            else:
                model.add(Conv2D(filters, (k, k), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))

        # flatten before Dense
        
        model.add(GlobalAveragePooling2D())

        # — Dense blocks —
        for neurons in list_number_of_nuerons_in_each_layer:
            model.add(Dense(neurons, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

    
        model.add(Dense(4, activation='softmax'))

        # compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'loss']
        )
        
        return model
