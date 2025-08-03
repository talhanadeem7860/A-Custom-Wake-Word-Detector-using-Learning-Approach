import tensorflow as tf
import numpy as np
import pyaudio
import librosa

# --- Configuration ---
MODEL_PATH = 'wake_word_model.h5'
SAMPLE_RATE = 16000
DURATION = 1  # seconds
CHUNK_SIZE = 1024
N_MFCC = 40
CONFIDENCE_THRESHOLD = 0.95 # Adjust this based on performance

# --- Load the Model ---
print("[INFO] Loading wake word model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Initialize Audio Stream ---
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

print("\n[INFO] Listening for wake word...")

# Buffer to hold audio chunks
audio_buffer = []

try:
    while True:
        # Read a chunk of audio data
        data = stream.read(CHUNK_SIZE)
        audio_chunk = np.frombuffer(data, dtype=np.float32)
        audio_buffer.extend(audio_chunk)

        # Keep the buffer at a fixed size (equal to the model's expected input duration)
        if len(audio_buffer) >= SAMPLE_RATE * DURATION:
            # Get the latest 1-second clip
            clip = np.array(audio_buffer[-(SAMPLE_RATE * DURATION):])

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=clip, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            
            # Reshape for model input
            mfccs = np.expand_dims(mfccs, axis=0) # Add batch dimension
            mfccs = np.expand_dims(mfccs, axis=-1) # Add channel dimension

            # Make a prediction
            prediction = model.predict(mfccs, verbose=0)[0][0]

            if prediction > CONFIDENCE_THRESHOLD:
                print(f"==> WAKE WORD DETECTED! (Confidence: {prediction:.2f})")
            
            # Slide the buffer window
            audio_buffer = audio_buffer[CHUNK_SIZE:]

except KeyboardInterrupt:
    print("\n[INFO] Stopping...")
finally:

    print("[INFO] Closing stream...")
    stream.stop_stream()
    stream.close()
    audio.terminate()