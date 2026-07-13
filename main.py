import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from skimage.metrics import structural_similarity as ssim
from models import build_autoencoder, build_fused_classifier
from utils import add_salt_and_pepper, apply_watershed, extract_texture_features

# 1. Paths & Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

# 2. Data Loading
raw_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels='inferred', label_mode='categorical',
    image_size=(100, 100), batch_size=32
)

# Capture metadata before mapping
class_names = raw_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")

# Normalize pixel values once
train_ds = raw_ds.map(lambda x, y: (x / 255.0, y))

# 3. Model Initialization
autoencoder = build_autoencoder()
classifier = build_fused_classifier(num_classes=num_classes, feat_shape=(2,))

# 4. Data Preparation for Training
# A. Autoencoder Pipeline (Noisy -> Clean)
autoencoder_ds = train_ds.map(lambda x, y: (add_salt_and_pepper(x), x))

# B. Classifier Pipeline ([Image, Features] -> Label)
def prepare_classifier_batch(image_batch, label_batch):
    features_batch = tf.map_fn(
        lambda img: tf.py_function(func=extract_texture_features, inp=[img], Tout=tf.float32),
        image_batch,
        fn_output_signature=tf.float32
    )
    features_batch.set_shape([None, 2])
    return ({"img_input": image_batch, "feat_input": features_batch}, label_batch)

classifier_train_ds = train_ds.map(prepare_classifier_batch)

# 5. Training
print("Training Autoencoder...")
autoencoder.fit(autoencoder_ds, epochs=3)

print("Training Fused Classifier...")
classifier.fit(classifier_train_ds, epochs=3)

# 6. Pipeline Execution (Testing)
def run_full_pipeline(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(100, 100))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    noisy = add_salt_and_pepper(img_array)
    
    restored = autoencoder.predict(np.expand_dims(noisy, 0), verbose=0)[0]
    score = ssim(img_array, restored, data_range=1.0, channel_axis=-1)
    
    segmented = apply_watershed(restored)
    tex_feats = extract_texture_features(segmented)
    
    pred = classifier.predict([np.expand_dims(restored, 0), tex_feats.reshape(1, -1)], verbose=0)
    return pred, score

print("\n--- Running Test Pipeline ---")
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            pred, ssim_score = run_full_pipeline(os.path.join(root, file))
            print(f"File: {file} | SSIM: {ssim_score:.4f} | Prediction: {class_names[np.argmax(pred)]}")