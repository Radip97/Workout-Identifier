"""
data_preprocessing.py — Video-to-Keypoints Preprocessing Pipeline
=================================================================
Scans the workout/ dataset directory, extracts MediaPipe Pose keypoints
from every video, normalises them, and saves the result as numpy arrays
ready for model training.

Outputs (in --output_dir):
  sequences.npz       → {'X': (N, SEQ_LEN, 132), 'y': (N,)}
  label2id.json       → {"BenchPress": 0, "BodyWeightSquats": 1, ...}
  id2label.json       → {"0": "BenchPress", "1": "BodyWeightSquats", ...}
  preprocessing_stats.json  → per-class counts and any skipped files

Usage:
  python data_preprocessing.py --data_dir workout --output_dir processed_data
  python data_preprocessing.py --data_dir workout --output_dir processed_data --seq_len 40 --skip_frames 2
"""

import os
import cv2
import json
import argparse
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from utils import normalize_keypoints, NUM_LANDMARKS, FEATURES_PER_LANDMARK

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEQ_LEN   = 30    # number of frames sampled per video
DEFAULT_SKIP_FRAMES = 1   # process every N-th frame (1 = all frames)
VIDEO_EXTENSIONS  = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# MediaPipe initialisation
# ---------------------------------------------------------------------------

mp_pose = mp.solutions.pose


def build_pose_extractor() -> mp.solutions.pose.Pose:
    """Create a reusable MediaPipe Pose instance."""
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def extract_keypoints_from_frame(
    frame: np.ndarray,
    pose: mp.solutions.pose.Pose,
) -> Optional[np.ndarray]:
    """
    Run MediaPipe on a single BGR OpenCV frame.

    Returns:
        1-D np.ndarray of shape (132,) if pose was detected, else None.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks is None:
        return None
    kps = []
    for lm in results.pose_landmarks.landmark:
        kps.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(kps, dtype=np.float32)


def sample_frames_uniformly(
    cap: cv2.VideoCapture,
    n_frames: int,
    skip: int = 1,
) -> List[np.ndarray]:
    """
    Read the video and return exactly *n_frames* BGR frames sampled uniformly.

    Args:
        cap:      OpenCV VideoCapture (already opened)
        n_frames: Number of frames to return
        skip:     Frame skip interval (process every *skip*-th frame)

    Returns:
        List of BGR np.ndarray frames
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: read all frames manually
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        total = len(frames)
    else:
        frames = None  # lazy — we will seek

    if total == 0:
        return []

    # Build target frame indices
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    indices = np.clip(indices, 0, total - 1)

    sampled = []
    if frames is not None:
        # Frames already in memory
        for idx in indices:
            sampled.append(frames[idx])
    else:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if ret:
                sampled.append(frame)
            else:
                # If seek fails, duplicate last valid frame
                if sampled:
                    sampled.append(sampled[-1])

    return sampled


def video_to_keypoint_sequence(
    video_path: str,
    pose: mp.solutions.pose.Pose,
    seq_len: int = DEFAULT_SEQ_LEN,
    skip: int = DEFAULT_SKIP_FRAMES,
) -> Optional[np.ndarray]:
    """
    Convert a single video to a normalised keypoint sequence.

    Args:
        video_path: Path to the video file
        pose:       Reusable MediaPipe Pose instance
        seq_len:    Target number of frames in output sequence
        skip:       Process every *skip* frames

    Returns:
        np.ndarray of shape (seq_len, 132) or None if keypoints cannot be extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return None

    frames = sample_frames_uniformly(cap, seq_len, skip)
    cap.release()

    if len(frames) == 0:
        print(f"  [WARN] No frames decoded: {video_path}")
        return None

    keypoint_sequence = []
    last_valid = None

    for frame in frames:
        kps = extract_keypoints_from_frame(frame, pose)
        if kps is not None:
            normalised = normalize_keypoints(kps)
            keypoint_sequence.append(normalised)
            last_valid = normalised
        else:
            # Use last valid frame or zeros to maintain sequence length
            keypoint_sequence.append(
                last_valid if last_valid is not None
                else np.zeros(NUM_LANDMARKS * FEATURES_PER_LANDMARK, dtype=np.float32)
            )

    # Ensure exactly seq_len frames
    if len(keypoint_sequence) < seq_len:
        pad = keypoint_sequence[-1] if keypoint_sequence else np.zeros(
            NUM_LANDMARKS * FEATURES_PER_LANDMARK, dtype=np.float32
        )
        while len(keypoint_sequence) < seq_len:
            keypoint_sequence.append(pad)
    elif len(keypoint_sequence) > seq_len:
        keypoint_sequence = keypoint_sequence[:seq_len]

    sequence = np.stack(keypoint_sequence, axis=0)  # (seq_len, 132)

    # Basic sanity check: reject if all zeros (pose never detected)
    if np.all(sequence == 0):
        return None

    return sequence

# ---------------------------------------------------------------------------
# Dataset scanning and processing
# ---------------------------------------------------------------------------

def scan_dataset(data_dir: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Scan the dataset directory and build a list of (video_path, class_label) pairs.

    Returns:
        videos:   List of (video_path, label) tuples
        label2id: Dict mapping label string → integer id
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    classes = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not classes:
        raise ValueError(f"No class subdirectories found in: {data_dir}")

    label2id = {cls: idx for idx, cls in enumerate(classes)}
    videos = []

    for cls in classes:
        cls_dir = data_path / cls
        for ext in VIDEO_EXTENSIONS:
            for video_file in cls_dir.glob(f"*{ext}"):
                videos.append((str(video_file), cls))
            for video_file in cls_dir.glob(f"*{ext.upper()}"):
                videos.append((str(video_file), cls))

    print(f"Found {len(classes)} classes: {classes}")
    print(f"Total videos found: {len(videos)}")
    return videos, label2id


def preprocess_dataset(
    data_dir: str,
    output_dir: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    skip_frames: int = DEFAULT_SKIP_FRAMES,
) -> None:
    """
    Main preprocessing pipeline:
    1. Scan video files
    2. Extract + normalise keypoints from each video
    3. Save sequences, labels, and mappings

    Args:
        data_dir:    Root of dataset (e.g., "workout")
        output_dir:  Where to save processed data
        seq_len:     Frames per sequence
        skip_frames: Frame skip interval
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    videos, label2id = scan_dataset(data_dir)
    id2label = {str(v): k for k, v in label2id.items()}

    # Save mappings
    with open(output_path / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(output_path / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"Saved label mappings to {output_path}/label2id.json")

    # Process videos
    pose = build_pose_extractor()
    all_sequences: List[np.ndarray] = []
    all_labels:   List[int] = []
    stats: Dict[str, Dict] = {cls: {"processed": 0, "skipped": 0} for cls in label2id}

    total = len(videos)
    for idx, (video_path, label) in enumerate(videos):
        print(f"  [{idx+1:4d}/{total}] {label} | {Path(video_path).name}", end=" ... ")

        seq = video_to_keypoint_sequence(video_path, pose, seq_len, skip_frames)

        if seq is None:
            print("SKIPPED (no pose detected)")
            stats[label]["skipped"] += 1
            continue

        all_sequences.append(seq)
        all_labels.append(label2id[label])
        stats[label]["processed"] += 1
        print(f"OK  shape={seq.shape}")

    pose.close()

    if not all_sequences:
        raise RuntimeError("No sequences were successfully processed. Check your dataset and MediaPipe installation.")

    X = np.stack(all_sequences, axis=0)  # (N, seq_len, 132)
    y = np.array(all_labels, dtype=np.int32)

    # Save sequences
    output_file = output_path / "sequences.npz"
    np.savez_compressed(str(output_file), X=X, y=y)
    print(f"\nSaved {X.shape[0]} sequences → {output_file}  (shape: {X.shape})")

    # Save statistics
    with open(output_path / "preprocessing_stats.json", "w") as f:
        json.dump({
            "total_videos_found":     total,
            "sequences_saved":        int(X.shape[0]),
            "seq_len":                seq_len,
            "frame_features":         X.shape[2],
            "classes":                label2id,
            "per_class_stats":        stats,
        }, f, indent=2)

    print("\n=== Preprocessing Complete ===")
    for cls, s in stats.items():
        print(f"  {cls:25s}  processed={s['processed']}  skipped={s['skipped']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe keypoints from workout videos."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="workout",
        help="Root directory of the dataset (default: workout)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Directory to save processed data (default: processed_data)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Number of frames per sequence (default: {DEFAULT_SEQ_LEN})"
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=DEFAULT_SKIP_FRAMES,
        help=f"Process every N-th frame (default: {DEFAULT_SKIP_FRAMES})"
    )
    args = parser.parse_args()

    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        skip_frames=args.skip_frames,
    )
