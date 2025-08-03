import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

# --- Configuration ---
WAKE_WORD = 'go'
DATASET_NAME = 'speech_commands'
SAMPLE_RATE = 16000
DURATION_S = 1.0 # All clips are 1 second
N_MFCC = 40 # Number of MFCCs to extract
FIXED_LENGTH = int(SAMPLE_RATE * DURATION_S)

# --- Data Loading and Preprocessing ---

# Find the integer label for our wake word
ds_builder = tfds.builder(DATASET_NAME)
label_names = ds_builder.info.features['label'].names
WAKE_WORD_LABEL = label_names.index(WAKE_WORD)
print(f"The wake word '{WAKE_WORD}' has label: {WAKE_WORD_LABEL}")

def preprocess(example):
    """Preprocesses a single audio clip from the dataset."""
    audio = example['audio']
    label = example['label']
    
    # Pad or truncate audio to the fixed length
    audio = tf.cast(audio, tf.float32)
    padding = FIXED_LENGTH - tf.shape(audio)[0]
    audio = tf.cond(padding > 0, lambda: tf.pad(audio, [[0, padding]]), lambda: audio[:FIXED_LENGTH])

    # Extract MFCCs using a tf.py_function to wrap librosa
    def extract_mfcc(audio_tensor):
        # Convert tensor to numpy array
        audio_np = audio_tensor.numpy()
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_np, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        return mfccs.astype(np.float32)

    mfccs = tf.py_function(func=extract_mfcc, inp=[audio], Tout=tf.float32)
    mfccs.set_shape([N_MFCC, None]) # Set shape for Keras
    
    # Create a binary label: 1 for wake word, 0 for everything else
    binary_label = tf.cast(label == WAKE_WORD_LABEL, tf.int64)
    
    return mfccs, binary_label

# Load the dataset
print("[INFO] Loading and preprocessing dataset...")
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    DATASET_NAME,
    split=['train', 'validation', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=False # We need the full example dict
)

# Apply preprocessing
train_dataset = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch the dataset for performance
BATCH_SIZE = 64
train_dataset = train_dataset.cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- Model Building (CNN) ---
print("[INFO] Building the CNN model...")

# Determine input shape from one sample
for mfccs, label in train_dataset.take(1):
    input_shape = mfccs.shape[1:]
print(f"Input shape for the model: {input_shape}")

model = Sequential([
    Input(shape=input_shape),
    # Reshape to add a channel dimension for Conv2D
    tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Train the Model ---
print("[INFO] Training the model...")
EPOCHS = 15
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --- Save the Model ---
print("[INFO] Saving the trained model...")
model.save('wake_word_model.h5')
print("[INFO] Training complete. Model saved as wake_word_model.h5")