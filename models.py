import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape=(100, 100, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def build_standard_classifier(num_classes, input_shape=(100, 100, 3), hp=None):
    """
    Standard CNN Classifier. Can be used normally or tuned via KerasTuner if 'hp' is provided.
    """
    # If KerasTuner is active, let it choose hyperparameters dynamically
    if hp is not None:
        filters_1 = hp.Choice('filters_1', values=[16, 32, 64])
        filters_2 = hp.Choice('filters_2', values=[32, 64, 128])
        dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    else:
        # Default fallback values
        filters_1 = 32
        filters_2 = 64
        dense_units = 128
        dropout_rate = 0.5
        lr = 0.001

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(filters_1, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters_2, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model