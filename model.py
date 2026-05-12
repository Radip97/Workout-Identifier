"""
model.py — TensorFlow/Keras Model Architecture
===============================================
Defines a Bidirectional LSTM + Temporal Attention model with two output heads:
  1. Classification head  → softmax over num_classes exercise labels
  2. Form quality head    → sigmoid scalar regression (0 = bad, 1 = perfect)

Configuration section at the top controls all hyperparameters.
The model is fully reusable after a single training run.

Usage (as a module):
    from model import build_model, load_model_from_path, CONFIG

    model = build_model(num_classes=7, seq_len=30, num_features=132)
    model.summary()
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# ⚙️  Configuration — edit hyperparameters here
# ---------------------------------------------------------------------------

CONFIG = {
    # --- Sequence ---
    "seq_len":       30,     # frames per input window (must match preprocessing)
    "num_features": 132,     # 33 landmarks × 4 (x, y, z, visibility)

    # --- LSTM layers ---
    "lstm_units":   [128, 64],   # units in each Bidirectional LSTM layer
    "dropout_rate":  0.35,

    # --- Dense layers before output ---
    "dense_units":  [128, 64],
    "l2_reg":        1e-4,

    # --- Attention ---
    "use_attention": True,   # temporal self-attention after LSTM stack

    # --- Output heads ---
    "use_form_head": True,   # regression head for form quality

    # --- Training (used by train.py) ---
    "batch_size":    32,
    "epochs":        80,
    "learning_rate": 1e-3,
    "lr_decay_patience": 8,
    "early_stop_patience": 20,
    "val_split":     0.2,
}

# ---------------------------------------------------------------------------
# Custom attention layer
# ---------------------------------------------------------------------------

class TemporalAttention(layers.Layer):
    """
    Soft attention over the time dimension.
    Learns to weight each timestep's contribution to the final representation.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False, activation="tanh")
        self.V = layers.Dense(1, use_bias=False)

    def call(self, hidden_seq: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            hidden_seq: (batch, timesteps, features)
        Returns:
            context:  (batch, features) — weighted sum
            weights:  (batch, timesteps, 1) — attention weights for visualisation
        """
        score = self.V(self.W(hidden_seq))           # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)        # (batch, T, 1)
        context = tf.reduce_sum(weights * hidden_seq, axis=1)  # (batch, features)
        return context, weights

    def get_config(self):
        config = super().get_config()
        return config

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int,
    seq_len:     int = CONFIG["seq_len"],
    num_features: int = CONFIG["num_features"],
    config: dict = None,
) -> Model:
    """
    Build and compile the Bidirectional LSTM exercise classifier.

    Architecture:
        Input → Normalisation → BiLSTM stack → Temporal Attention
              → Dense block → [Classification head | Form quality head]

    Args:
        num_classes:  Number of exercise classes
        seq_len:      Frames per sequence
        num_features: Feature vector size per frame (default 132)
        config:       Override CONFIG dict (optional)

    Returns:
        Compiled Keras Model
    """
    cfg = config or CONFIG

    # ── Input ────────────────────────────────────────────────────────────
    inputs = keras.Input(shape=(seq_len, num_features), name="keypoints_input")

    # ── Layer normalisation (per-timestep) ───────────────────────────────
    x = layers.LayerNormalization(name="layer_norm")(inputs)

    # ── Bidirectional LSTM stack ─────────────────────────────────────────
    for i, units in enumerate(cfg["lstm_units"]):
        return_seq = True  # always return sequences (attention needs full seq)
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=return_seq,
                dropout=cfg["dropout_rate"] * 0.5,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizers.l2(cfg["l2_reg"]),
            ),
            name=f"bilstm_{i+1}",
        )(x)
        x = layers.Dropout(cfg["dropout_rate"], name=f"dropout_lstm_{i+1}")(x)

    # ── Temporal attention ───────────────────────────────────────────────
    if cfg.get("use_attention", True):
        attn_layer = TemporalAttention(units=64, name="temporal_attention")
        x, _ = attn_layer(x)
    else:
        # Fallback: global average pool over time
        x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Shared dense block ───────────────────────────────────────────────
    for i, units in enumerate(cfg["dense_units"]):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(cfg["l2_reg"]),
            name=f"dense_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(cfg["dropout_rate"], name=f"dropout_dense_{i+1}")(x)

    # ── Classification head ──────────────────────────────────────────────
    class_output = layers.Dense(
        num_classes, activation="softmax", name="class_output"
    )(x)

    # ── Form quality head (optional regression) ──────────────────────────
    outputs = [class_output]
    loss    = {"class_output": "sparse_categorical_crossentropy"}
    metrics = {"class_output": ["accuracy"]}
    loss_weights = {"class_output": 1.0}

    if cfg.get("use_form_head", True):
        form_output = layers.Dense(
            32, activation="relu", name="form_dense"
        )(x)
        form_output = layers.Dense(
            1, activation="sigmoid", name="form_output"
        )(form_output)
        outputs.append(form_output)
        loss["form_output"]         = "mse"
        metrics["form_output"]      = ["mae"]
        loss_weights["form_output"] = 0.3   # secondary task

    # ── Compile ──────────────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="exercise_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    return model


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_model_from_path(model_path: str) -> Model:
    """
    Load a previously saved Keras model.

    Handles the custom TemporalAttention layer transparently.

    Args:
        model_path: Path to the saved .h5 or SavedModel directory

    Returns:
        Loaded Keras Model ready for inference
    """
    custom_objects = {"TemporalAttention": TemporalAttention}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model loaded from: {model_path}")
    return model


def save_model(model: Model, path: str) -> None:
    """Save model to .h5 file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.save(path)
    print(f"Model saved to: {path}")


# ---------------------------------------------------------------------------
# Quick model summary entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Print architecture summary with 7-class example
    m = build_model(num_classes=7, seq_len=CONFIG["seq_len"])
    m.summary(line_length=90)
    print(f"\nInput shape:  {m.input_shape}")
    print(f"Output shapes: {[o.shape for o in m.outputs]}")
    print(f"\nConfig: {json.dumps(CONFIG, indent=2)}")
