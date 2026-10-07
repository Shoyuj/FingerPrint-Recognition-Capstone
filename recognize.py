"""
Fingerprint Recognition & Verification CLI Script
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from model import load_trained_model, label_to_description, label_to_key
from data_generator import load_dataset, get_default_augmenter


def parse_args():
    parser = argparse.ArgumentParser(description='Fingerprint Recognition & Verification')
    parser.add_argument('--mode', type=str, default='verify', choices=['verify', 'search', 'eval'],
                        help='Mode: verify (1:1 match demo), search (1:N DB search), eval (test set metrics)')
    parser.add_argument('--model_path', type=str, default='result/fingerprint_model.keras',
                        help='Path to trained model')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='result/verification_result.png',
                        help='Path to save output visualization')
    parser.add_argument('--num_tests', type=int, default=5, help='Number of test pairs/queries to evaluate')
    return parser.parse_args()


def run_verification_demo(model, data, output_path='result/verification_result.png', num_samples=3):
    """
    Runs 1:1 Siamese verification demo on random altered query fingerprints vs true & false matches.
    """
    x_real = data['x_real']
    y_real = data['y_real']
    x_test = data['x_hard']
    y_test = data['y_hard']
    label_real_dict = data['label_real_dict']
    augmenter = get_default_augmenter()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    print("\n" + "=" * 60)
    print(" 1:1 FINGERPRINT VERIFICATION RESULTS ")
    print("=" * 60)

    for row in range(num_samples):
        # Pick random test fingerprint
        rand_idx = random.randint(0, len(x_test) - 1)
        query_raw = x_test[rand_idx]
        query_label = y_test[rand_idx]
        match_key = label_to_key(query_label)

        # Apply augmentation to simulate live scan noise
        query_aug = augmenter.augment_image(query_raw)
        query_input = (query_aug.reshape((1, 90, 90, 1)).astype(np.float32) / 255.0)

        # Genuine database match
        true_idx = label_real_dict.get(match_key)
        if true_idx is None:
            continue
        rx = x_real[true_idx]
        ry = y_real[true_idx]
        rx_input = (rx.reshape((1, 90, 90, 1)).astype(np.float32) / 255.0)

        # Imposter database fingerprint
        while True:
            unmatch_key, unmatch_idx = random.choice(list(label_real_dict.items()))
            if unmatch_key != match_key:
                break
        ux = x_real[unmatch_idx]
        uy = y_real[unmatch_idx]
        ux_input = (ux.reshape((1, 90, 90, 1)).astype(np.float32) / 255.0)

        # Model inference
        pred_match = float(model.predict([query_input, rx_input], verbose=0)[0][0])
        pred_unmatch = float(model.predict([query_input, ux_input], verbose=0)[0][0])

        print(f"\n[Test #{row + 1}] Query: {label_to_description(query_label)}")
        print(f"  -> Genuine Target  ({label_to_key(ry)}): Confidence = {pred_match * 100:.2f}% | Verdict: {'MATCH (O)' if pred_match >= 0.5 else 'MISMATCH'}")
        print(f"  -> Imposter Target ({label_to_key(uy)}): Confidence = {pred_unmatch * 100:.2f}% | Verdict: {'MISMATCH (X)' if pred_unmatch < 0.5 else 'FALSE MATCH'}")

        # Plot Query
        axes[row, 0].imshow(query_aug.squeeze(), cmap='gray')
        axes[row, 0].set_title(f"Input Query\nKey: {match_key}\n{label_to_description(query_label)}", fontsize=9)
        axes[row, 0].axis('off')

        # Plot Genuine Match
        axes[row, 1].imshow(rx.squeeze(), cmap='gray')
        color = 'green' if pred_match >= 0.5 else 'red'
        axes[row, 1].set_title(f"Genuine Match: {pred_match * 100:.1f}%\nKey: {label_to_key(ry)}\n{label_to_description(ry)}",
                               fontsize=9, color=color, fontweight='bold')
        axes[row, 1].axis('off')

        # Plot Imposter Mismatch
        axes[row, 2].imshow(ux.squeeze(), cmap='gray')
        color = 'red' if pred_unmatch >= 0.5 else 'black'
        axes[row, 2].set_title(f"Imposter Target: {pred_unmatch * 100:.1f}%\nKey: {label_to_key(uy)}\n{label_to_description(uy)}",
                               fontsize=9, color=color)
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("\n" + "=" * 60)
    print(f"Verification visualization saved to {output_path}")


def run_database_search(model, data, num_queries=3, top_k=5):
    """
    Runs 1:N Biometric Identification: searches query fingerprint across the real database.
    """
    x_real = data['x_real']
    y_real = data['y_real']
    x_test = data['x_medium']
    y_test = data['y_medium']
    augmenter = get_default_augmenter()

    print("\n" + "=" * 60)
    print(f" 1:N BIOMETRIC DATABASE IDENTIFICATION SEARCH (N={len(x_real)}) ")
    print("=" * 60)

    # Prepare normalized database in batches
    x_real_norm = x_real.astype(np.float32) / 255.0

    for q_idx in range(num_queries):
        rand_idx = random.randint(0, len(x_test) - 1)
        query_raw = x_test[rand_idx]
        query_label = y_test[rand_idx]
        query_key = label_to_key(query_label)

        query_aug = augmenter.augment_image(query_raw)
        query_input = (query_aug.reshape((1, 90, 90, 1)).astype(np.float32) / 255.0)

        # Tile query for all database items
        query_tiled = np.repeat(query_input, len(x_real), axis=0)

        # Predict similarities across entire database
        preds = model.predict([query_tiled, x_real_norm], batch_size=256, verbose=0).flatten()

        # Top-K indices
        top_indices = np.argsort(preds)[::-1][:top_k]

        print(f"\n[Query #{q_idx + 1}] Target Key: {query_key} ({label_to_description(query_label)})")
        print(f"  Top {top_k} Candidates from Database:")
        for rank, db_idx in enumerate(top_indices, 1):
            cand_label = y_real[db_idx]
            cand_key = label_to_key(cand_label)
            score = preds[db_idx] * 100
            is_correct = (cand_key == query_key)
            mark = "✓ [CORRECT]" if is_correct else "✗"
            print(f"    Rank {rank}: Key={cand_key} | Confidence={score:5.1f}% | {mark} | {label_to_description(cand_label)}")


def run_evaluation_metrics(model, data, test_samples=500):
    """
    Evaluates ROC metrics, precision, recall, and overall test accuracy on balanced pairs.
    """
    x_real = data['x_real']
    x_test = data['x_hard']
    y_test = data['y_hard']
    label_real_dict = data['label_real_dict']
    keys = list(label_real_dict.keys())

    print("\n" + "=" * 60)
    print(f" EVALUATING TEST METRICS ({test_samples * 2} pairs) ")
    print("=" * 60)

    pairs_x1 = []
    pairs_x2 = []
    y_true = []

    indices = np.random.choice(len(x_test), test_samples, replace=False)

    for idx in indices:
        q_img = x_test[idx]
        q_label = y_test[idx]
        match_key = label_to_key(q_label)

        if match_key in label_real_dict:
            # Positive pair
            pairs_x1.append(q_img)
            pairs_x2.append(x_real[label_real_dict[match_key]])
            y_true.append(1.0)

            # Negative pair
            while True:
                unmatch_key = random.choice(keys)
                if unmatch_key != match_key:
                    break
            pairs_x1.append(q_img)
            pairs_x2.append(x_real[label_real_dict[unmatch_key]])
            y_true.append(0.0)

    pairs_x1 = np.array(pairs_x1).astype(np.float32) / 255.0
    pairs_x2 = np.array(pairs_x2).astype(np.float32) / 255.0
    y_true = np.array(y_true)

    preds = model.predict([pairs_x1, pairs_x2], batch_size=64, verbose=1).flatten()
    y_pred_binary = (preds >= 0.5).astype(np.float32)

    acc = np.mean(y_pred_binary == y_true)
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    print("\n" + "-" * 40)
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    print(f"  Precision:     {precision * 100:.2f}%")
    print(f"  Recall:        {recall * 100:.2f}%")
    print(f"  F1 Score:      {f1 * 100:.2f}%")
    print(f"  True Positives: {tp} | False Positives: {fp}")
    print(f"  True Negatives: {tn} | False Negatives: {fn}")
    print("-" * 40)


def main():
    args = parse_args()
    print(f"Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path)
    print("Loading dataset...")
    data = load_dataset(args.dataset_dir)

    if args.mode == 'verify':
        run_verification_demo(model, data, output_path=args.output_path, num_samples=args.num_tests)
    elif args.mode == 'search':
        run_database_search(model, data, num_queries=args.num_tests)
    elif args.mode == 'eval':
        run_evaluation_metrics(model, data)


if __name__ == '__main__':
    main()
