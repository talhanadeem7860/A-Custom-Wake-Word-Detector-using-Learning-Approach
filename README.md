A Custom Wake Word Detector using a Learning Approach

Project Overview

This repository contains the implementation of a custom "wake word" detector using a deep learning approach. The system is designed to listen to an audio stream in real-time and recognize a specific keyword, demonstrating a complete audio machine learning pipeline from raw signal processing to a live inference application.

The model is trained to recognize the word "go" as the positive wake word, distinguishing it from thousands of other words and background noises sourced from the Google Speech Commands dataset.

Methodology

The project is executed in two main phases:

Data Processing and Model Training:

The project utilizes the Google Speech Commands dataset, automatically managed by the TensorFlow Datasets library. Audio clips are processed into a machine-understandable format by extracting Mel-Frequency Cepstral Coefficients (MFCCs). A Convolutional Neural Network (CNN) is then trained on these features to learn the acoustic differences between the designated wake word and other sounds.

Real-Time Detection:

A real-time listening application accesses the system's microphone using the PyAudio library, capturing live audio in short chunks. It continuously processes these chunks to extract MFCCs and feeds them into the trained CNN model. When the model predicts the wake word with high confidence, a confirmation message is triggered.

Functionality

The final application listens passively through the microphone. Upon recognizing the keyword "go," it prints a "WAKE WORD DETECTED!" message to the console, demonstrating a successful real-time audio classification.
