"""
inference.py — Model Inference Engine
======================================
Provides a clean API for running predictions using the saved model.
Supports:
  - Single keypoint sequence prediction
  - Full video file prediction
  - Batch video inference from CLI

The inference pipeline loads the model ONCE and reuses it indefinitely —
no retraining required.

Usage (as a module):
    from inference import load_pipeline, predict_sequence, predict_video

    pipeline = load_pipeline("saved_models")
    label, confidence, form_score = predict_sequence(pipeline, my_sequence)

Usage (CLI):
    python inference.py --model_dir saved_models --video path/to/video.avi
    python inference.py --model_dir saved_models --video_dir workout/PushUps
"""

import os
import json
import pickle
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mediapipe as mp

from model import load_model_from_path
from data_preprocessing import (
    build_pose_extractor,
    extract_keypoints_from_frame,
    sample_frames_uniformly,
    DEFAULT_SEQ_LEN,
)
from utils import normalize_keypoints, NUM_LANDMARKS, FEATURES_PER_LANDMARK

# ---------------------------------------------------------------------------
# Pipeline container
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Holds all inference artefacts loaded from the saved_models directory.

    Attributes:
        model:    Loaded Keras model
        label2id: {"BenchPress": 0, ...}
        id2label: {"0": "BenchPress", ...}
        scaler:   Fitted sklearn StandardScaler (or None)
        seq_len:  Sequence length expected by the model
        config:   Training config dict
    """

    def __init__(self, model, label2id, id2label, scaler, seq_len, config):
        self.model    = model
        self.label2id = label2id
        self.id2label = id2label
        self.scaler   = scaler
        self.seq_len  = seq_len
        self.config   = config
        self.num_features = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 132

    def scale(self, X: np.ndarray) -> np.ndarray:
        """Scale (N, T, F) or (T, F) array using the saved scaler."""
        if self.scaler is None:
            return X
        single = X.ndim == 2
        if single:
            X = X[np.newaxis]       # (1, T, F)
        N, T, F = X.shape
        X_scaled = self.scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
        return X_scaled[0] if single else X_scaled


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(model_dir: str) -> Pipeline:
    """
    Load all inference artefacts from a saved_models directory.

    Args:
        model_dir: directory containing model.h5, label2id.json, scaler.pkl, config.json

    Returns:
        Pipeline instance ready for inference
    """
    model_dir = Path(model_dir)

    # Try model.h5, then model_best.h5
    for model_name in ("model.h5", "model_best.h5"):
        model_path = model_dir / model_name
        if model_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No model.h5 or model_best.h5 found in '{model_dir}'. "
            "Run train.py first."
        )

    model = load_model_from_path(str(model_path))

    # Label mappings
    label2id_path = model_dir / "label2id.json"
    id2label_path = model_dir / "id2label.json"
    with open(label2id_path) as f:
        label2id = json.load(f)
    with open(id2label_path) as f:
        id2label = json.load(f)

    # Scaler (optional)
    scaler_path = model_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded.")

    # Config (optional)
    config_path = model_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    seq_len = config.get("seq_len", DEFAULT_SEQ_LEN)

    print(f"Pipeline ready — classes: {list(label2id.keys())}")
    return Pipeline(model, label2id, id2label, scaler, seq_len, config)


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_sequence(
    pipeline: Pipeline,
    sequence: np.ndarray,
) -> Tuple[str, float, float]:
    """
    Run inference on a single keypoint sequence.

    Args:
        pipeline: Loaded Pipeline object
        sequence: np.ndarray of shape (T, 132) — normalised keypoints

    Returns:
        (label, confidence, form_score)
        - label:      Predicted exercise class name
        - confidence: Softmax probability [0, 1]
        - form_score: Form quality estimate [0, 1]
    """
    if sequence.shape[0] != pipeline.seq_len:
        # Resize to expected length via interpolation
        from scipy.interpolate import interp1d
        old_t = np.linspace(0, 1, sequence.shape[0])
        new_t = np.linspace(0, 1, pipeline.seq_len)
        f = interp1d(old_t, sequence, axis=0, kind="linear")
        sequence = f(new_t).astype(np.float32)

    X = sequence[np.newaxis]           # (1, T, 132)
    X = pipeline.scale(X)

    preds = pipeline.model.predict(X, verbose=0)

    if isinstance(preds, list):
        class_probs  = preds[0][0]     # (num_classes,)
        form_score   = float(preds[1][0][0]) if len(preds) > 1 else 0.5
    else:
        class_probs = preds[0]
        form_score  = 0.5

    class_id   = int(np.argmax(class_probs))
    confidence = float(class_probs[class_id])
    label      = pipeline.id2label[str(class_id)]

    return label, confidence, form_score


def predict_video(
    pipeline: Pipeline,
    video_path: str,
    stride: int = 15,
) -> List[Dict]:
    """
    Run sliding-window inference over a full video file.

    Args:
        pipeline:   Loaded Pipeline
        video_path: Path to the video file
        stride:     Frames to advance the window each step

    Returns:
        List of dicts, one per window:
        [{"label": str, "confidence": float, "form_score": float, "window_start": int}, ...]
    """
    pose = build_pose_extractor()
    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose.close()
        raise IOError(f"Cannot open video: {video_path}")

    print(f"Processing video: {video_path}")

    # Collect all keypoints from the video
    all_kps: List[Optional[np.ndarray]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kps = extract_keypoints_from_frame(frame, pose)
        if kps is not None:
            all_kps.append(normalize_keypoints(kps))
        else:
            all_kps.append(
                all_kps[-1] if all_kps else np.zeros(pipeline.num_features, dtype=np.float32)
            )
        frame_idx += 1

    cap.release()
    pose.close()

    if len(all_kps) < pipeline.seq_len:
        print(f"  Video too short ({len(all_kps)} frames < {pipeline.seq_len} needed).")
        return []

    results = []
    seq_arr = np.stack(all_kps, axis=0)  # (total_frames, 132)

    for start in range(0, len(seq_arr) - pipeline.seq_len + 1, stride):
        window = seq_arr[start: start + pipeline.seq_len]
        label, conf, form = predict_sequence(pipeline, window)
        results.append({
            "label":        label,
            "confidence":   conf,
            "form_score":   form,
            "window_start": start,
        })
        print(f"  Window [{start:4d}-{start+pipeline.seq_len-1:4d}]  "
              f"{label:25s}  conf={conf:.2%}  form={form:.2f}")

    # Aggregate: majority vote
    if results:
        from collections import Counter
        votes = Counter(r["label"] for r in results)
        best_label, vote_count = votes.most_common(1)[0]
        avg_conf = np.mean([r["confidence"] for r in results if r["label"] == best_label])
        avg_form = np.mean([r["form_score"] for r in results])
        print(f"\n  Majority prediction: {best_label}  "
              f"({vote_count}/{len(results)} windows)  "
              f"avg_conf={avg_conf:.2%}  avg_form={avg_form:.2f}")

    return results


# ---------------------------------------------------------------------------
# CLI — batch video inference
# ---------------------------------------------------------------------------

def batch_infer_directory(
    pipeline: Pipeline,
    video_dir: str,
    output_json: Optional[str] = None,
    recursive: bool = True,
) -> List[Dict]:
    """
    Run inference on every video in a directory and optionally save to JSON.

    If recursive=True (default), scans all subdirectories too — so you can
    point it at the root "workout" folder and process ALL classes at once.
    The ground-truth label is inferred from the parent folder name.
    """
    video_dir = Path(video_dir)
    extensions = ("*.avi", "*.mp4", "*.mov", "*.mkv")

    video_files = []
    for ext in extensions:
        if recursive:
            video_files.extend(video_dir.rglob(ext))
        else:
            video_files.extend(video_dir.glob(ext))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return []

    print(f"Found {len(video_files)} video(s) to process.\n")

    all_results = []
    correct = 0
    total   = 0

    for vf in sorted(video_files):
        per_video = predict_video(pipeline, str(vf))
        if per_video:
            from collections import Counter
            votes = Counter(r["label"] for r in per_video)
            best  = votes.most_common(1)[0][0]
            avg_c = np.mean([r["confidence"] for r in per_video if r["label"] == best])

            # Ground truth = parent folder name
            ground_truth = vf.parent.name

            is_correct = (best == ground_truth)
            if ground_truth in pipeline.label2id:
                total += 1
                if is_correct:
                    correct += 1

            all_results.append({
                "file":             str(vf.relative_to(video_dir)),
                "ground_truth":     ground_truth,
                "predicted_label":  best,
                "correct":          is_correct,
                "avg_confidence":   float(avg_c),
                "windows":          per_video,
            })

    # Print accuracy summary
    if total > 0:
        print(f"\n{'='*60}")
        print(f"  OVERALL ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"{'='*60}\n")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved batch results to {output_json}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exercise inference on video files.")
    parser.add_argument("--model_dir", type=str, default="saved_models",
                        help="Directory with trained model artefacts (default: saved_models)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Alias: path like saved_models/model.h5")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory of videos — recursively scans subfolders")
    parser.add_argument("--output", type=str, default=None,
                        help="Save batch results to this JSON file")
    parser.add_argument("--stride", type=int, default=15,
                        help="Window stride in frames (default: 15)")
    parser.add_argument("--no_recursive", action="store_true",
                        help="Disable recursive scanning of subdirectories")
    args = parser.parse_args()

    # Resolve model directory
    model_dir = args.model_dir
    if args.model_path:
        model_dir = os.path.dirname(args.model_path) or model_dir

    pipeline = load_pipeline(model_dir)

    if args.video:
        predict_video(pipeline, args.video, stride=args.stride)
    elif args.video_dir:
        batch_infer_directory(
            pipeline, args.video_dir,
            output_json=args.output,
            recursive=not args.no_recursive,
        )
    else:
        print("Provide --video or --video_dir to run inference.")
        print("Examples:")
        print("  python inference.py --model_dir saved_models --video workout/PushUps/v_PushUps_g01_c01.avi")
        print("  python inference.py --model_dir saved_models --video_dir workout --output all_results.json")
