import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)



class model_size:
    def get_model_size_mb(self, model):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            model.save(tmp_file.name, save_format='h5')
            size_in_mb = os.path.getsize(tmp_file.name) / (1024 * 1024)
            os.remove(tmp_file.name)
            return size_in_mb



if __name__ == "__main__": 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input

    # Build a simple dummy model
    def create_dummy_model():
        model = Sequential([
            Input(shape=(28, 28, 1)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Create the model
    dummy_model = create_dummy_model()

    # Test model size function
    #from packages.return_model_size import model_size

    ms = model_size()
    size = ms.get_model_size_mb(dummy_model)

    print(f"Model size: {size:.2f} MB")