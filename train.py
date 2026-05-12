"""
train.py — Model Training Pipeline
====================================
Loads preprocessed keypoint sequences, trains the Bidirectional LSTM model,
and saves all artefacts needed for reusable inference.

Saved artefacts (in --model_dir):
  model.h5              → Trained Keras model
  label2id.json         → {"BenchPress": 0, ...}
  id2label.json         → {"0": "BenchPress", ...}
  scaler.pkl            → sklearn StandardScaler fitted on training data
  training_history.json → Loss / accuracy per epoch
  confusion_matrix.png  → Validation confusion matrix
  training_curves.png   → Loss and accuracy plots

Usage:
  python train.py --data_dir processed_data --model_dir saved_models
  python train.py --data_dir processed_data --model_dir saved_models --epochs 100 --batch_size 16
"""

import os
import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

from model import build_model, save_model, CONFIG

# Reproducibility
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------

def augment_sequence(seq: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Light temporal augmentation to reduce overfitting.
    - Adds small Gaussian noise to x,y,z coordinates.
    - Randomly flips the sequence horizontally with 50% probability.

    Args:
        seq:       (T, 132) normalised keypoint sequence
        noise_std: standard deviation of Gaussian noise

    Returns:
        Augmented (T, 132) sequence
    """
    augmented = seq.copy()

    # Gaussian noise on x, y, z channels (every 4th is visibility — skip)
    noise = np.zeros_like(augmented)
    noise[:, 0::4] = np.random.normal(0, noise_std, noise[:, 0::4].shape)  # x
    noise[:, 1::4] = np.random.normal(0, noise_std, noise[:, 1::4].shape)  # y
    noise[:, 2::4] = np.random.normal(0, noise_std * 0.3, noise[:, 2::4].shape)  # z (smaller)
    augmented += noise

    # Horizontal flip: negate x coordinates (left-right mirror)
    if np.random.rand() < 0.5:
        augmented[:, 0::4] *= -1

    return augmented


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    aug_factor: int = 2,
) -> tuple:
    """
    Duplicate the dataset *aug_factor* times with augmentation.

    Returns:
        (X_aug, y_aug) with shape[0] == X.shape[0] * aug_factor
    """
    X_parts = [X]
    y_parts = [y]
    for _ in range(aug_factor - 1):
        X_aug = np.stack([augment_sequence(seq) for seq in X], axis=0)
        X_parts.append(X_aug)
        y_parts.append(y)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_preprocessed_data(data_dir: str):
    """
    Load sequences.npz and label mappings produced by data_preprocessing.py.

    Args:
        data_dir: path to processed_data directory

    Returns:
        X: (N, T, F) float32
        y: (N,)   int32
        label2id: dict
        id2label: dict
    """
    data_path = Path(data_dir)
    sequences_file = data_path / "sequences.npz"
    label2id_file  = data_path / "label2id.json"
    id2label_file  = data_path / "id2label.json"

    if not sequences_file.exists():
        raise FileNotFoundError(
            f"sequences.npz not found in '{data_dir}'. "
            "Run data_preprocessing.py first."
        )

    data = np.load(str(sequences_file))
    X = data["X"].astype(np.float32)  # (N, T, 132)
    y = data["y"].astype(np.int32)    # (N,)

    with open(label2id_file) as f:
        label2id = json.load(f)
    with open(id2label_file) as f:
        id2label = json.load(f)

    print(f"Loaded {X.shape[0]} sequences  shape={X.shape}  classes={len(label2id)}")
    return X, y, label2id, id2label

# ---------------------------------------------------------------------------
# Scaler (fit on training data, applies per-feature across time axis)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on the training split.
    Flattens time and samples to compute per-feature mean/std.
    """
    N, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Transform (N, T, F) with a pre-fitted scaler."""
    N, T, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)

# ---------------------------------------------------------------------------
# Synthetic form labels (pseudo-labels for the form regression head)
# ---------------------------------------------------------------------------

def generate_synthetic_form_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Generate pseudo form-quality labels to supervise the regression head.
    Without a labelled form-quality dataset we use random values [0.6-1.0]
    to stabilise the head; the real discriminative learning will come from
    the classification loss.

    Replace with real labels if available.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.uniform(0.6, 1.0, size=len(y)).astype(np.float32)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(model_dir: str, config: dict) -> list:
    """Build training callbacks: model checkpoint, early stop, LR schedule."""
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "model_best.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_class_output_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_class_output_accuracy",
            patience=config["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config["lr_decay_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "tb_logs"),
            histogram_freq=0,
        ),
    ]
    return callbacks

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_dir: str) -> None:
    """Save training / validation loss and accuracy curves."""
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14)

    # Loss
    axes[0].plot(epochs, hist.get("loss", []),     label="Train Loss")
    axes[0].plot(epochs, hist.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Classification accuracy
    acc_key  = "class_output_accuracy"
    val_key  = "val_class_output_accuracy"
    if acc_key in hist:
        axes[1].plot(epochs, hist[acc_key],     label="Train Acc")
        axes[1].plot(epochs, hist.get(val_key, []), label="Val Acc")
        axes[1].set_title("Classification Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(
    model, X_val: np.ndarray, y_val: np.ndarray, id2label: dict, output_dir: str
) -> None:
    """Compute and save the validation confusion matrix."""
    preds = model.predict(X_val, verbose=0)
    # preds may be a list (two heads) or a single array
    if isinstance(preds, list):
        y_pred = np.argmax(preds[0], axis=1)
    else:
        y_pred = np.argmax(preds, axis=1)

    class_names = [id2label[str(i)] for i in range(len(id2label))]
    cm = confusion_matrix(y_val, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=30, ha="right")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")

# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(
    data_dir:   str = "processed_data",
    model_dir:  str = "saved_models",
    epochs:     int = CONFIG["epochs"],
    batch_size: int = CONFIG["batch_size"],
    augment:    bool = True,
    aug_factor: int = 2,
) -> None:
    """
    Full training pipeline:
      1. Load data
      2. Scale features
      3. Augment training set
      4. Build model
      5. Train with callbacks
      6. Save model + artefacts
      7. Plot curves and confusion matrix

    Args:
        data_dir:   Directory with preprocessed sequences
        model_dir:  Directory to save trained model and artefacts
        epochs:     Maximum training epochs
        batch_size: Mini-batch size
        augment:    Whether to apply data augmentation
        aug_factor: How many times to duplicate training data with augmentation
    """
    os.makedirs(model_dir, exist_ok=True)

    # --- 1. Load ---
    X, y, label2id, id2label = load_preprocessed_data(data_dir)
    num_classes = len(label2id)
    _, seq_len, num_features = X.shape

    # Per-class sample count
    print("\nPer-class distribution:")
    for cls, cid in label2id.items():
        print(f"  {cls:28s}  {np.sum(y == cid):4d} samples")

    # --- 2. Train/val split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=CONFIG["val_split"],
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\nSplit → train: {len(X_train)}  val: {len(X_val)}")

    # --- 3. Fit scaler on training data, transform both splits ---
    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    # --- 4. Data augmentation ---
    if augment:
        X_train, y_train = augment_dataset(X_train, y_train, aug_factor=aug_factor)
        # Shuffle
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        print(f"After augmentation (×{aug_factor}): {len(X_train)} training samples")

    # --- 5. Build model ---
    model = build_model(num_classes=num_classes, seq_len=seq_len, num_features=num_features)
    model.summary(line_length=90)

    # --- 6. Prepare targets ---
    use_form_head = CONFIG.get("use_form_head", True)
    if use_form_head:
        form_train = generate_synthetic_form_labels(y_train, num_classes)
        form_val   = generate_synthetic_form_labels(y_val,   num_classes)
        train_targets = {"class_output": y_train, "form_output": form_train}
        val_targets   = {"class_output": y_val,   "form_output": form_val}
    else:
        train_targets = {"class_output": y_train}
        val_targets   = {"class_output": y_val}

    # --- 7. Train ---
    callbacks = build_callbacks(model_dir, CONFIG)
    config_for_run = dict(CONFIG, epochs=epochs, batch_size=batch_size)

    print(f"\nStarting training — {epochs} epochs, batch_size={batch_size}")
    history = model.fit(
        X_train,
        train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, val_targets),
        callbacks=callbacks,
        verbose=1,
    )

    # --- 8. Save final model and artefacts ---
    final_model_path = os.path.join(model_dir, "model.h5")
    save_model(model, final_model_path)

    # Copy label mappings into model_dir
    import shutil
    for fname in ("label2id.json", "id2label.json"):
        src = os.path.join(data_dir, fname)
        dst = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname} → {dst}")

    # Save config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_for_run, f, indent=2)

    # Save training history
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # --- 9. Plots ---
    plot_training_curves(history, model_dir)
    plot_confusion_matrix(model, X_val, y_val, id2label, model_dir)

    # --- 10. Final val accuracy ---
    val_preds = model.predict(X_val, verbose=0)
    if isinstance(val_preds, list):
        val_acc = np.mean(np.argmax(val_preds[0], axis=1) == y_val)
    else:
        val_acc = np.mean(np.argmax(val_preds, axis=1) == y_val)
    print(f"\n✓ Final validation accuracy: {val_acc*100:.2f}%")
    print(f"✓ All artefacts saved to:    {os.path.abspath(model_dir)}/")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the exercise classification model."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed_data",
        help="Directory with preprocessed sequences (default: processed_data)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory to save trained model (default: saved_models)"
    )
    # Legacy alias for compatibility with the spec
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Ignored if --model_dir is set; path like saved_models/model.h5"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["epochs"],
        help=f"Training epochs (default: {CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CONFIG["batch_size"],
        help=f"Mini-batch size (default: {CONFIG['batch_size']})"
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable training data augmentation"
    )
    parser.add_argument(
        "--aug_factor",
        type=int,
        default=2,
        help="Augmentation multiplier (default: 2)"
    )
    args = parser.parse_args()

    # Resolve model_dir from --model_path if needed
    model_dir = args.model_dir
    if args.model_path and model_dir == "saved_models":
        model_dir = os.path.dirname(args.model_path) or "saved_models"

    train(
        data_dir=args.data_dir,
        model_dir=model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        aug_factor=args.aug_factor,
    )
