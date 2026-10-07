"""
Dataset Loader and Data Generator for Fingerprint Recognition
"""

import os
import random
import numpy as np
import keras
from imgaug import augmenters as iaa
from model import label_to_key


def load_dataset(dataset_dir='dataset'):
    """
    Load real and altered fingerprint datasets and return them as numpy arrays.
    """
    x_real = np.load(os.path.join(dataset_dir, 'x_real.npz'))['data']
    y_real = np.load(os.path.join(dataset_dir, 'y_real.npy'))

    x_easy = np.load(os.path.join(dataset_dir, 'x_easy.npz'))['data']
    y_easy = np.load(os.path.join(dataset_dir, 'y_easy.npy'))

    x_medium = np.load(os.path.join(dataset_dir, 'x_medium.npz'))['data']
    y_medium = np.load(os.path.join(dataset_dir, 'y_medium.npy'))

    x_hard = np.load(os.path.join(dataset_dir, 'x_hard.npz'))['data']
    y_hard = np.load(os.path.join(dataset_dir, 'y_hard.npy'))

    # Build lookup map for real fingerprints by key
    label_real_dict = {}
    for i, y in enumerate(y_real):
        key = label_to_key(y)
        label_real_dict[key] = i

    return {
        'x_real': x_real,
        'y_real': y_real,
        'x_easy': x_easy,
        'y_easy': y_easy,
        'x_medium': x_medium,
        'y_medium': y_medium,
        'x_hard': x_hard,
        'y_hard': y_hard,
        'label_real_dict': label_real_dict
    }


def get_default_augmenter():
    """
    Returns the standard image augmentation pipeline for altered fingerprints.
    """
    return iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=255
        )
    ], random_order=True)


class FingerprintDataGenerator(keras.utils.PyDataset):
    """
    Batch generator for Siamese CNN training and validation.
    Generates pairs of (x1, x2) and target label (1.0 for match, 0.0 for mismatch).
    """
    def __init__(self, x, labels, x_real, label_real_dict, batch_size=32, shuffle=True, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.labels = labels
        self.x_real = x_real
        self.label_real_dict = label_real_dict
        self.label_keys = list(label_real_dict.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seq = get_default_augmenter() if augment else None

        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x1_batch = self.x[batch_indices].copy()
        labels_batch = self.labels[batch_indices]

        x2_batch = np.empty((self.batch_size, 90, 90, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, 1), dtype=np.float32)

        # Apply augmentation to input images
        if self.augment and self.seq is not None:
            x1_batch = self.seq.augment_images(x1_batch)

        for i, l in enumerate(labels_batch):
            match_key = label_to_key(l)

            # 50% chance of positive match, 50% chance of negative match
            if random.random() > 0.5:
                # Positive pair (same subject and finger)
                if match_key in self.label_real_dict:
                    real_idx = self.label_real_dict[match_key]
                    x2_batch[i] = self.x_real[real_idx]
                    y_batch[i] = 1.0
                else:
                    # Fallback if key missing
                    rand_key = random.choice(self.label_keys)
                    x2_batch[i] = self.x_real[self.label_real_dict[rand_key]]
                    y_batch[i] = 0.0
            else:
                # Negative pair (different subject/finger)
                while True:
                    unmatch_key = random.choice(self.label_keys)
                    if unmatch_key != match_key:
                        break
                x2_batch[i] = self.x_real[self.label_real_dict[unmatch_key]]
                y_batch[i] = 0.0

        # Normalize pixel values to [0, 1]
        x1_norm = (x1_batch.astype(np.float32) / 255.0)
        x2_norm = (x2_batch.astype(np.float32) / 255.0)

        return (x1_norm, x2_norm), y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
