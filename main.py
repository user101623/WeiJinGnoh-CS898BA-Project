import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt
from models import build_standard_classifier

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

# Split train_ds into training and validation subsets for Hyperparameter Tuning
val_size = int(0.2 * tf.data.experimental.cardinality(train_ds).numpy())
val_ds = train_ds.take(val_size)
tune_train_ds = train_ds.skip(val_size)

# 3. Hyperparameter Tuning for the Classifier using KerasTuner
print("\nRunning Hyperparameter Optimization for Classifier...")

def model_builder(hp):
    return build_standard_classifier(num_classes=num_classes, hp=hp)

tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='cnn_tuning'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

tuner.search(
    tune_train_ds, 
    epochs=5, 
    validation_data=val_ds, 
    callbacks=[stop_early]
)

# Retrieve the best tuned classifier model
classifier = tuner.get_best_models(num_models=1)[0]
print("Hyperparameter tuning completed successfully!")

# 4. Pipeline Execution 
print("\n--- Running Test Pipeline ---")

total_images = 0
correct_predictions = 0

for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(('.jpg')):
            total_images += 1
            img_path = os.path.join(root, file)
            
            # Extract true label from the parent folder name
            true_label = os.path.basename(root)
            
            # Run prediction
            img = tf.keras.utils.load_img(img_path, target_size=(100, 100))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            pred = classifier.predict(np.expand_dims(img_array, 0), verbose=0)
            
            predicted_label = class_names[np.argmax(pred)]
            
            # Check if correct
            is_correct = (predicted_label.lower() == true_label.lower())
            if is_correct:
                correct_predictions += 1
                
            status = "CORRECT" if is_correct else f"WRONG (True: {true_label})"
            print(f"File: {file} | Prediction: {predicted_label} | Status: {status}")

# Print final test accuracy if test data was structured in folders
if total_images > 0:
    test_accuracy = correct_predictions / total_images
    print(f"\n==============================")
    print(f"Total Test Images: {total_images}")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"==============================")
else:
    print("\nNo images found. Make sure your Test directory contains subfolders named after the classes.")