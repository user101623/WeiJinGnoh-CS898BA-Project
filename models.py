import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape=(100, 100, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # Added padding='same'
    
    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    
    # Map the 32 filters back to 3 RGB channels
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Slower, safer step
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )

    return model

def build_fused_classifier(num_classes, img_shape=(100, 100, 3), feat_shape=(2,)):
    # Spatial Path (CNN)
    img_input = layers.Input(shape=img_shape, name="img_input")
    x = layers.Conv2D(32, (3, 3), activation='relu')(img_input)
    x = layers.MaxPooling2D()(x)
    spatial = layers.GlobalAveragePooling2D()(x)
    
    # Feature Path (Manual Texture)
    feat_input = layers.Input(shape=feat_shape, name="feat_input")
    
    # Fusion
    combined = layers.concatenate([spatial, feat_input])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.BatchNormalization()(z)
    
    # Final Classification Layer
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    # Single Model definition
    model = models.Model(inputs=[img_input, feat_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model