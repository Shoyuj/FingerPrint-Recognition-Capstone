"""
Training Script for Siamese CNN Fingerprint Recognition
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_siamese_model, label_to_description, label_to_key
from data_generator import load_dataset, FingerprintDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train Siamese CNN for Fingerprint Recognition')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--quick', action='store_true', help='Quick training mode on a smaller subset (for testing)')
    parser.add_argument('--subset_size', type=int, default=10000, help='Number of samples in quick mode')
    parser.add_argument('--save_model', type=str, default='result/fingerprint_model.keras', help='Path to save trained model')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='result', help='Path to output directory')
    return parser.parse_args()


def plot_and_save_metrics(history, output_path='result/training_curves.png'):
    """
    Plot training & validation loss and accuracy curves and save to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs_range = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(epochs_range, history.history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    plt.plot(epochs_range, history.history[acc_key], 'b-o', label='Training Accuracy')
    if val_acc_key in history.history:
        plt.plot(epochs_range, history.history[val_acc_key], 'r-s', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training metrics plot saved to {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(" Fingerprint Recognition Siamese CNN Training ")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Batch size: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Model output path: {args.save_model}")

    # 1. Load datasets
    print("\n[1/4] Loading datasets...")
    data = load_dataset(args.dataset_dir)
    x_real = data['x_real']
    y_real = data['y_real']
    x_easy = data['x_easy']
    y_easy = data['y_easy']
    x_medium = data['x_medium']
    y_medium = data['y_medium']
    x_hard = data['x_hard']
    y_hard = data['y_hard']
    label_real_dict = data['label_real_dict']

    print(f"  Real database fingerprints: {x_real.shape[0]} images")
    print(f"  Altered Easy: {x_easy.shape[0]} | Medium: {x_medium.shape[0]} | Hard: {x_hard.shape[0]}")

    # Combine altered datasets
    x_all = np.concatenate([x_easy, x_medium, x_hard], axis=0)
    labels_all = np.concatenate([y_easy, y_medium, y_hard], axis=0)

    if args.quick:
        print(f"\n[Quick Mode] Subsampling dataset to {args.subset_size} samples...")
        sample_indices = np.random.choice(len(x_all), min(args.subset_size, len(x_all)), replace=False)
        x_all = x_all[sample_indices]
        labels_all = labels_all[sample_indices]

    print(f"  Total training candidates: {len(x_all)} images")

    # Split into train & validation sets
    x_train, x_val, label_train, label_val = train_test_split(
        x_all, labels_all, test_size=args.val_split, random_state=42
    )
    print(f"  Train samples: {len(x_train)} | Val samples: {len(x_val)}")

    # 2. Build Data Generators
    print("\n[2/4] Initializing Siamese Data Generators...")
    train_gen = FingerprintDataGenerator(
        x_train, label_train, x_real, label_real_dict,
        batch_size=args.batch_size, shuffle=True, augment=True
    )
    val_gen = FingerprintDataGenerator(
        x_val, label_val, x_real, label_real_dict,
        batch_size=args.batch_size, shuffle=False, augment=False
    )

    # 3. Build Model
    print("\n[3/4] Building Siamese CNN Model...")
    model, feature_extractor = build_siamese_model(
        input_shape=(90, 90, 1), learning_rate=args.lr
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=args.save_model,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 4. Train Model
    print(f"\n[4/4] Starting training for {args.epochs} epochs...")
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")

    # Save model
    model.save(args.save_model)
    print(f"Final model saved to {args.save_model}")

    # Plot metrics
    plot_and_save_metrics(history, os.path.join(args.output_dir, 'training_curves.png'))

    # Quick evaluation
    val_eval = model.evaluate(val_gen, verbose=1)
    print("\n" + "=" * 60)
    print(f"Final Evaluation on Validation Set:")
    print(f"  Loss:     {val_eval[0]:.4f}")
    print(f"  Accuracy: {val_eval[1] * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
