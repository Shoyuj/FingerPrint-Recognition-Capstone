"""
Siamese CNN Model Architecture for Fingerprint Recognition
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import keras
from keras import layers, models


FINGER_NAMES = {
    0: 'Thumb',
    1: 'Index',
    2: 'Middle',
    3: 'Ring',
    4: 'Little'
}

GENDER_NAMES = {
    0: 'Male',
    1: 'Female'
}

HAND_NAMES = {
    0: 'Left Hand',
    1: 'Right Hand'
}


def label_to_key(label):
    """
    Convert a 4-element label array [subject_id, gender, lr, finger] to a 6-digit key string.
    e.g., [122, 0, 0, 1] -> '122001'
    """
    subject_id = int(label[0])
    gender = int(label[1])
    lr = int(label[2])
    finger = int(label[3])
    return f"{subject_id:03d}{gender}{lr}{finger}"


def label_to_description(label):
    """
    Convert label to readable text.
    """
    subject_id = int(label[0])
    gender = GENDER_NAMES.get(int(label[1]), 'Unknown')
    hand = HAND_NAMES.get(int(label[2]), 'Unknown')
    finger = FINGER_NAMES.get(int(label[3]), 'Unknown')
    return f"Subject {subject_id} ({gender}) - {hand}, {finger} Finger"


def build_siamese_model(input_shape=(90, 90, 1), learning_rate=1e-3):
    """
    Builds Siamese CNN network for fingerprint verification.
    """
    # Twin inputs
    x1 = layers.Input(shape=input_shape, name='input_fingerprint_1')
    x2 = layers.Input(shape=input_shape, name='input_fingerprint_2')

    # Shared feature extractor CNN
    feat_in = layers.Input(shape=input_shape, name='feature_input')
    feat = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', name='conv1')(feat_in)
    feat = layers.MaxPooling2D(pool_size=2, name='pool1')(feat)
    feat = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', name='conv2')(feat)
    feat = layers.MaxPooling2D(pool_size=2, name='pool2')(feat)

    feature_extractor = models.Model(inputs=feat_in, outputs=feat, name='shared_feature_extractor')

    # Extract features from both inputs
    x1_feat = feature_extractor(x1)
    x2_feat = feature_extractor(x2)

    # Compute difference between feature representations
    diff = layers.Subtract(name='feature_subtraction')([x1_feat, x2_feat])

    # Classification / similarity network
    net = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', name='diff_conv')(diff)
    net = layers.MaxPooling2D(pool_size=2, name='diff_pool')(net)
    net = layers.Flatten(name='flatten')(net)
    net = layers.Dense(64, activation='relu', name='dense_1')(net)
    output = layers.Dense(1, activation='sigmoid', name='match_probability')(net)

    model = models.Model(inputs=[x1, x2], outputs=output, name='Siamese_Fingerprint_Verifier')

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model, feature_extractor


def load_trained_model(model_path='result/fingerprint_model.keras'):
    """
    Load a trained Siamese fingerprint model from disk.
    """
    if not os.path.exists(model_path):
        # Check if .h5 variant exists
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    model = keras.models.load_model(model_path)
    return model
