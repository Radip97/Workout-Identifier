"""
utils.py — Shared utilities for the Exercise Classification System
==================================================================
Contains:
  - Angle and vector math for joint-angle computation
  - Keypoint normalization (hip-centred, scale-invariant)
  - Sliding-window buffer used by the real-time app
  - Per-exercise form-evaluation functions
  - OpenCV/HUD drawing helpers

MediaPipe Pose landmark index reference (used throughout):
  0  NOSE                 11  LEFT_SHOULDER   12  RIGHT_SHOULDER
  13 LEFT_ELBOW           14  RIGHT_ELBOW
  15 LEFT_WRIST           16  RIGHT_WRIST
  23 LEFT_HIP             24  RIGHT_HIP
  25 LEFT_KNEE            26  RIGHT_KNEE
  27 LEFT_ANKLE           28  RIGHT_ANKLE
  29 LEFT_HEEL            30  RIGHT_HEEL
  11 LEFT_SHOULDER        12  RIGHT_SHOULDER
  (full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
"""

import numpy as np
import cv2
import math
from collections import deque
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of MediaPipe Pose landmarks
NUM_LANDMARKS = 33
# Features per landmark (x, y, z, visibility)
FEATURES_PER_LANDMARK = 4
# Total feature vector size per frame
FRAME_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 132

# MediaPipe landmark indices
LM = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,  "LEFT_EYE": 2,  "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,  "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,     "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

# Exercise display colours (BGR)
EXERCISE_COLORS = {
    "BenchPress":           (0, 200, 255),
    "BodyWeightSquats":     (50, 255, 100),
    "Lunges":               (255, 150, 50),
    "PullUps":              (200, 50, 255),
    "PushUps":              (50, 200, 255),
    "barbell biceps curl":  (255, 220, 50),
    "hammer curl":          (50, 255, 220),
}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_landmark_xyz(frame_keypoints: np.ndarray, landmark_idx: int) -> np.ndarray:
    """
    Extract (x, y, z) from a flat frame keypoint vector.

    Args:
        frame_keypoints: 1-D array of length FRAME_FEATURES (132)
        landmark_idx:    MediaPipe landmark index (0-32)

    Returns:
        np.ndarray of shape (3,) → [x, y, z]
    """
    base = landmark_idx * FEATURES_PER_LANDMARK
    return frame_keypoints[base:base + 3]


def get_landmark_visibility(frame_keypoints: np.ndarray, landmark_idx: int) -> float:
    """Return visibility score for a given landmark."""
    base = landmark_idx * FEATURES_PER_LANDMARK
    return float(frame_keypoints[base + 3])


def angle_between_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Compute the angle (degrees) at point *b* formed by the vectors b→a and b→c.

    Args:
        a, b, c: (x, y) or (x, y, z) coordinate arrays

    Returns:
        Angle in degrees [0, 180].
    """
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def compute_joint_angle(
    frame_kps: np.ndarray,
    proximal: int,
    joint: int,
    distal: int,
) -> float:
    """
    Convenience wrapper: angle (°) at *joint* landmark between proximal and distal.
    """
    a = get_landmark_xyz(frame_kps, proximal)
    b = get_landmark_xyz(frame_kps, joint)
    c = get_landmark_xyz(frame_kps, distal)
    return angle_between_points(a, b, c)


def torso_height(frame_kps: np.ndarray) -> float:
    """Approximate torso height: distance from mid-hip to mid-shoulder."""
    mid_hip = 0.5 * (
        get_landmark_xyz(frame_kps, LM["LEFT_HIP"]) +
        get_landmark_xyz(frame_kps, LM["RIGHT_HIP"])
    )
    mid_shoulder = 0.5 * (
        get_landmark_xyz(frame_kps, LM["LEFT_SHOULDER"]) +
        get_landmark_xyz(frame_kps, LM["RIGHT_SHOULDER"])
    )
    height = np.linalg.norm(mid_shoulder[:2] - mid_hip[:2])
    return max(height, 1e-6)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_keypoints(frame_kps: np.ndarray) -> np.ndarray:
    """
    Hip-centred, torso-scale normalisation applied to a single frame.

    Steps:
      1. Translate so mid-hip is the origin.
      2. Scale by the torso height to make it scale-invariant.
      3. Visibility scores are kept but not scaled.

    Args:
        frame_kps: 1-D np.ndarray of shape (132,)

    Returns:
        Normalised 1-D np.ndarray of shape (132,)
    """
    kps = frame_kps.copy().astype(np.float32).reshape(NUM_LANDMARKS, FEATURES_PER_LANDMARK)

    mid_hip = 0.5 * (kps[LM["LEFT_HIP"], :3] + kps[LM["RIGHT_HIP"], :3])
    kps[:, :3] -= mid_hip  # translate

    scale = torso_height(frame_kps)
    kps[:, :3] /= scale    # scale

    return kps.flatten()


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply hip-centred normalisation to every frame in a sequence.

    Args:
        sequence: np.ndarray of shape (T, 132)

    Returns:
        np.ndarray of shape (T, 132)
    """
    return np.stack([normalize_keypoints(frame) for frame in sequence], axis=0)

# ---------------------------------------------------------------------------
# Sliding-window buffer
# ---------------------------------------------------------------------------

class SlidingWindowBuffer:
    """
    Fixed-length FIFO buffer for streaming keypoint frames.

    Usage:
        buf = SlidingWindowBuffer(seq_len=30)
        buf.add(frame_kps)          # add one frame (132,)
        if buf.is_ready():
            seq = buf.get_sequence()  # (30, 132) ready for model
    """

    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        self._buffer: deque = deque(maxlen=seq_len)

    def add(self, frame_kps: np.ndarray) -> None:
        """Append one frame keypoint vector."""
        self._buffer.append(frame_kps.astype(np.float32))

    def is_ready(self) -> bool:
        """True once the buffer has been filled at least once."""
        return len(self._buffer) == self.seq_len

    def get_sequence(self) -> np.ndarray:
        """Return the current window as (seq_len, FRAME_FEATURES)."""
        return np.stack(list(self._buffer), axis=0)

    def reset(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

# ---------------------------------------------------------------------------
# Rep counting helpers
# ---------------------------------------------------------------------------

class RepCounter:
    """
    Threshold-based joint-angle rep counter.

    Counts a repetition each time the angle crosses *low_thresh* (down phase)
    and then recovers above *high_thresh* (up phase).
    """

    def __init__(
        self,
        low_thresh: float = 90.0,
        high_thresh: float = 160.0,
    ):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self._in_rep = False  # currently in the "down" phase
        self._count = 0

    def update(self, angle: float) -> int:
        """
        Feed the latest joint angle and return the updated rep count.
        """
        if not self._in_rep and angle <= self.low_thresh:
            self._in_rep = True  # entered down phase
        elif self._in_rep and angle >= self.high_thresh:
            self._in_rep = False  # completed a rep
            self._count += 1
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._in_rep = False
        self._count = 0

# ---------------------------------------------------------------------------
# Per-exercise form evaluation
# ---------------------------------------------------------------------------

def evaluate_form_benchpress(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate bench-press form from a sequence of keypoints.

    Rules:
      - Elbow angle should reach < 90° at the bottom of the movement.
      - Wrists should stay above the elbows throughout.

    Args:
        sequence: np.ndarray of shape (T, 132)

    Returns:
        (score [0-1], feedback_text, is_perfect)
    """
    feedback = []
    penalties = 0.0

    left_elbow_angles = []
    right_elbow_angles = []

    for frame in sequence:
        left_elbow_angles.append(
            compute_joint_angle(frame, LM["LEFT_SHOULDER"], LM["LEFT_ELBOW"], LM["LEFT_WRIST"])
        )
        right_elbow_angles.append(
            compute_joint_angle(frame, LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"])
        )

    min_left  = min(left_elbow_angles)
    min_right = min(right_elbow_angles)
    min_elbow = min(min_left, min_right)

    if min_elbow > 100:
        feedback.append("Lower the bar further — elbow angle too shallow")
        penalties += 0.4
    elif min_elbow > 90:
        feedback.append("Good depth, try to reach 90° elbow angle")
        penalties += 0.15

    # Wrist-above-elbow check (y-coordinate; smaller y = higher on screen)
    wrist_above_count = 0
    for frame in sequence:
        lw_y = get_landmark_xyz(frame, LM["LEFT_WRIST"])[1]
        le_y = get_landmark_xyz(frame, LM["LEFT_ELBOW"])[1]
        if lw_y < le_y:  # wrist higher than elbow
            wrist_above_count += 1
    if wrist_above_count / max(len(sequence), 1) < 0.5:
        feedback.append("Keep wrists directly above elbows")
        penalties += 0.2

    score = max(0.0, 1.0 - penalties)
    is_perfect = score >= 0.85 and not feedback
    if not feedback:
        feedback.append("Excellent bench press form!")
    return score, " | ".join(feedback), is_perfect


def evaluate_form_bodyweightsquats(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate bodyweight squat form.

    Rules:
      - Knee angle must reach < 100° (deep squat).
      - Knee tracking: knee should not cave inward significantly.
      - Back angle: torso should be relatively upright.
    """
    penalties = 0.0
    feedback = []

    knee_angles = []
    for frame in sequence:
        left_knee  = compute_joint_angle(frame, LM["LEFT_HIP"],  LM["LEFT_KNEE"],  LM["LEFT_ANKLE"])
        right_knee = compute_joint_angle(frame, LM["RIGHT_HIP"], LM["RIGHT_KNEE"], LM["RIGHT_ANKLE"])
        knee_angles.append(min(left_knee, right_knee))

    min_knee = min(knee_angles)

    if min_knee > 110:
        feedback.append("Squat deeper — aim for thighs parallel to floor")
        penalties += 0.4
    elif min_knee > 100:
        feedback.append("Good squat! Try to go a touch lower")
        penalties += 0.15

    # Knee cave detection: knee x should not move significantly inward vs. hip
    cave_count = 0
    for frame in sequence:
        left_knee_x  = get_landmark_xyz(frame, LM["LEFT_KNEE"])[0]
        left_hip_x   = get_landmark_xyz(frame, LM["LEFT_HIP"])[0]
        left_ankle_x = get_landmark_xyz(frame, LM["LEFT_ANKLE"])[0]
        # knee should be roughly between hip and ankle x
        if left_knee_x > left_hip_x + 0.1:
            cave_count += 1
    if cave_count / max(len(sequence), 1) > 0.3:
        feedback.append("Watch knee caves — push knees out over toes")
        penalties += 0.2

    score = max(0.0, 1.0 - penalties)
    is_perfect = score >= 0.85 and not feedback
    if not feedback:
        feedback.append("Perfect squat form!")
    return score, " | ".join(feedback), is_perfect


def evaluate_form_lunges(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate lunge form.

    Rules:
      - Front knee angle ~90° at lowest point.
      - Torso remains upright (shoulder over hip).
    """
    penalties = 0.0
    feedback = []

    knee_angles = []
    torso_angles = []

    for frame in sequence:
        # Use minimum knee angle across both sides
        lk = compute_joint_angle(frame, LM["LEFT_HIP"],  LM["LEFT_KNEE"],  LM["LEFT_ANKLE"])
        rk = compute_joint_angle(frame, LM["RIGHT_HIP"], LM["RIGHT_KNEE"], LM["RIGHT_ANKLE"])
        knee_angles.append(min(lk, rk))

        # Torso angle: angle between shoulder-hip and vertical
        shoulder = get_landmark_xyz(frame, LM["LEFT_SHOULDER"])
        hip      = get_landmark_xyz(frame, LM["LEFT_HIP"])
        vertical = np.array([hip[0], hip[1] - 1.0])  # point directly above hip
        torso_angles.append(angle_between_points(shoulder, hip, vertical))

    min_knee = min(knee_angles)
    mean_torso = np.mean(torso_angles)

    if min_knee > 110:
        feedback.append("Step further — achieve 90° knee bend")
        penalties += 0.35
    elif min_knee > 100:
        feedback.append("Almost there — aim for 90° front knee angle")
        penalties += 0.1

    if mean_torso > 25:
        feedback.append("Keep your torso upright during the lunge")
        penalties += 0.25

    score = max(0.0, 1.0 - penalties)
    is_perfect = score >= 0.85 and not feedback
    if not feedback:
        feedback.append("Great lunge form!")
    return score, " | ".join(feedback), is_perfect


def evaluate_form_pullups(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate pull-up form.

    Rules:
      - Full elbow extension at bottom (>160°).
      - Chin above elbow at top (head y < elbow y in normalized coords).
      - Kipping: minimal swing (hip x should not oscillate excessively).
    """
    penalties = 0.0
    feedback = []

    max_elbow = 0.0
    min_elbow = 180.0
    hip_x_vals = []

    for frame in sequence:
        le = compute_joint_angle(frame, LM["LEFT_SHOULDER"], LM["LEFT_ELBOW"], LM["LEFT_WRIST"])
        re = compute_joint_angle(frame, LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"])
        avg = (le + re) / 2.0
        max_elbow = max(max_elbow, avg)
        min_elbow = min(min_elbow, avg)
        hip_x_vals.append(get_landmark_xyz(frame, LM["LEFT_HIP"])[0])

    if max_elbow < 150:
        feedback.append("Fully extend arms at the bottom of each rep")
        penalties += 0.35
    if min_elbow > 70:
        feedback.append("Pull higher — elbows should come fully down")
        penalties += 0.3

    # Kipping: standard deviation of hip x
    if len(hip_x_vals) > 1 and np.std(hip_x_vals) > 0.15:
        feedback.append("Reduce kipping — focus on strict pull-ups")
        penalties += 0.25

    score = max(0.0, 1.0 - penalties)
    is_perfect = score >= 0.85 and not feedback
    if not feedback:
        feedback.append("Excellent pull-up form!")
    return score, " | ".join(feedback), is_perfect


def evaluate_form_pushups(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate push-up form.

    Rules:
      - Elbow angle < 90° at bottom.
      - Body straight: hip should not sag (hip y should stay near shoulder line).
      - Elbows at ~45° from body (not too flared).
    """
    penalties = 0.0
    feedback = []

    elbow_angles = []
    hip_sag_vals = []

    for frame in sequence:
        le = compute_joint_angle(frame, LM["LEFT_SHOULDER"], LM["LEFT_ELBOW"], LM["LEFT_WRIST"])
        re = compute_joint_angle(frame, LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"])
        elbow_angles.append(min(le, re))

        # Hip sag: hip y vs. shoulder y and ankle y interpolated line
        shoulder_y = get_landmark_xyz(frame, LM["LEFT_SHOULDER"])[1]
        hip_y      = get_landmark_xyz(frame, LM["LEFT_HIP"])[1]
        ankle_y    = get_landmark_xyz(frame, LM["LEFT_ANKLE"])[1]
        # Expected hip y if body is straight: linearly interpolated
        if abs(ankle_y - shoulder_y) > 1e-6:
            expected_hip_y = shoulder_y + 0.5 * (ankle_y - shoulder_y)
        else:
            expected_hip_y = shoulder_y
        hip_sag_vals.append(hip_y - expected_hip_y)  # positive = hip dropped below line

    min_elbow = min(elbow_angles)
    if min_elbow > 100:
        feedback.append("Lower your chest — elbow angle too shallow")
        penalties += 0.4
    elif min_elbow > 90:
        feedback.append("Good! Try to lower chest just a bit more")
        penalties += 0.1

    mean_sag = np.mean(hip_sag_vals)
    if mean_sag > 0.1:
        feedback.append("Hips are sagging — engage your core")
        penalties += 0.35
    elif mean_sag < -0.1:
        feedback.append("Hips too high — keep body in a straight line")
        penalties += 0.2

    score = max(0.0, 1.0 - penalties)
    is_perfect = score >= 0.85 and not feedback
    if not feedback:
        feedback.append("Excellent push-up form!")
    return score, " | ".join(feedback), is_perfect


def evaluate_form_barbell_biceps_curl(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate barbell biceps curl form.

    Rules:
      - Full range of motion: elbow should reach ~30-40° at top and ~160° at bottom.
      - Elbow drift: elbows should not swing forward excessively.
    """
    penalties = 0.0
    feedback = []

    elbow_angles = []
    elbow_drift = []

    for frame in sequence:
        le = compute_joint_angle(frame, LM["LEFT_SHOULDER"], LM["LEFT_ELBOW"], LM["LEFT_WRIST"])
        re = compute_joint_angle(frame, LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"])
        elbow_angles.append((le + re) / 2.0)

        # Elbow drift: elbow x vs. shoulder x (elbow should stay close to body)
        lelbow_x    = get_landmark_xyz(frame, LM["LEFT_ELBOW"])[0]
        lshoulder_x = get_landmark_xyz(frame, LM["LEFT_SHOULDER"])[0]
        elbow_drift.append(abs(lelbow_x - lshoulder_x))

    min_elbow = min(elbow_angles)
    max_elbow = max(elbow_angles)

    if min_elbow > 55:
        feedback.append("Squeeze harder at the top — contract your biceps fully")
        penalties += 0.25
    if max_elbow < 150:
        feedback.append("Fully extend at the bottom for full range of motion")
        penalties += 0.25

    mean_drift = np.mean(elbow_drift)
    if mean_drift > 0.12:
        feedback.append("Elbows drifting — keep them pinned to your sides")
        penalties += 0.3

    score = max(0.0, 1.0 - penalties)
    is_perfect = score >= 0.85 and not feedback
    if not feedback:
        feedback.append("Perfect curl form!")
    return score, " | ".join(feedback), is_perfect


def evaluate_form_hammer_curl(
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Evaluate hammer curl form.

    Same rules as barbell curl; additionally checks wrist stays neutral.
    """
    # Re-use the same rules as barbell curl for the structural checks
    score, feedback_text, is_perfect = evaluate_form_barbell_biceps_curl(sequence)
    # Downgrade slightly if feedback is generic "Perfect" because hammer curls
    # are typically stricter about wrist rotation (hard to detect without IMU)
    return score, feedback_text.replace("curl form", "hammer curl form"), is_perfect


# ---------------------------------------------------------------------------
# Form evaluation dispatcher
# ---------------------------------------------------------------------------

EXERCISE_FORM_EVALUATORS = {
    "BenchPress":           evaluate_form_benchpress,
    "BodyWeightSquats":     evaluate_form_bodyweightsquats,
    "Lunges":               evaluate_form_lunges,
    "PullUps":              evaluate_form_pullups,
    "PushUps":              evaluate_form_pushups,
    "barbell biceps curl":  evaluate_form_barbell_biceps_curl,
    "hammer curl":          evaluate_form_hammer_curl,
}


def evaluate_form(
    exercise_name: str,
    sequence: np.ndarray,
) -> Tuple[float, str, bool]:
    """
    Dispatch form evaluation for the given exercise.

    Args:
        exercise_name: Label string (must match a key in EXERCISE_FORM_EVALUATORS)
        sequence: np.ndarray (T, 132) of normalised keypoints

    Returns:
        (score [0-1], feedback_text, is_perfect)
    """
    evaluator = EXERCISE_FORM_EVALUATORS.get(exercise_name)
    if evaluator is None:
        return 0.5, f"No form rules defined for '{exercise_name}'", False
    return evaluator(sequence)


# ---------------------------------------------------------------------------
# Rep-counter configuration per exercise
# ---------------------------------------------------------------------------

def get_rep_angle_config(exercise_name: str) -> Dict:
    """
    Return the landmark triplets and thresholds for rep counting.

    Returns dict with keys: joints (list of triplets), combine, low_thresh, high_thresh
    """
    def bilateral(p1, j1, d1, p2, j2, d2):
        return [(LM[p1], LM[j1], LM[d1]), (LM[p2], LM[j2], LM[d2])]

    configs = {
        "BenchPress": dict(
            joints=bilateral("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            combine="min", low_thresh=90, high_thresh=160,
        ),
        "BodyWeightSquats": dict(
            joints=bilateral("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            combine="min", low_thresh=100, high_thresh=160,
        ),
        "Lunges": dict(
            joints=bilateral("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            combine="min", low_thresh=100, high_thresh=160,
        ),
        "PullUps": dict(
            joints=bilateral("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            combine="min", low_thresh=70, high_thresh=155,
        ),
        "PushUps": dict(
            joints=bilateral("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            combine="min", low_thresh=90, high_thresh=160,
        ),
        "barbell biceps curl": dict(
            joints=bilateral("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            combine="min", low_thresh=50, high_thresh=155,
        ),
        "hammer curl": dict(
            joints=bilateral("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            combine="min", low_thresh=50, high_thresh=155,
        ),
    }
    return configs.get(exercise_name, dict(
        joints=[(LM["LEFT_SHOULDER"], LM["LEFT_ELBOW"], LM["LEFT_WRIST"])],
        combine="min", low_thresh=90, high_thresh=160,
    ))

# ---------------------------------------------------------------------------
# HUD Drawing helpers
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    exercise: str,
    confidence: float,
    rep_count: int,
    form_score: float,
    feedback: str,
    is_perfect: bool,
    angle: Optional[float] = None,
) -> np.ndarray:
    """
    Render a rich semi-transparent HUD overlay on the given frame.

    Args:
        frame:       BGR image (H, W, 3)
        exercise:    Predicted exercise label
        confidence:  Model confidence [0, 1]
        rep_count:   Current rep count
        form_score:  Form quality [0, 1]
        feedback:    Form feedback text
        is_perfect:  Whether form is perfect
        angle:       Optional: current key joint angle to display

    Returns:
        Annotated BGR frame
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # --- Left panel background ---
    panel_w = min(420, w // 2)
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color = EXERCISE_COLORS.get(exercise, (200, 200, 200))

    y = 40
    # Exercise name
    _draw_text(frame, exercise.upper(), (20, y), scale=0.85, color=color, thickness=2)

    # Confidence bar
    y += 45
    _draw_label_and_bar(frame, "Confidence", confidence, (20, y), w=panel_w - 40, color=color)

    # Form quality bar
    y += 45
    form_color = _score_color(form_score)
    _draw_label_and_bar(frame, "Form Quality", form_score, (20, y), w=panel_w - 40, color=form_color)

    # Rep count
    y += 55
    _draw_text(frame, f"REPS: {rep_count}", (20, y), scale=1.2, color=(255, 255, 255), thickness=3)

    # Joint angle
    if angle is not None:
        y += 50
        _draw_text(frame, f"Angle: {angle:.1f}\u00b0", (20, y), scale=0.7, color=(200, 200, 200))

    # Perfect form badge
    if is_perfect:
        y += 50
        badge_text = "\u2605 PERFECT FORM \u2605"
        _draw_text(frame, badge_text, (20, y), scale=0.75,
                   color=(50, 255, 150), thickness=2)

    # Feedback text (multi-line wrap)
    y += 55
    _draw_text(frame, "Form tip:", (20, y), scale=0.55, color=(160, 160, 160))
    y += 25
    for line in _wrap_text(feedback, max_chars=48):
        _draw_text(frame, line, (20, y), scale=0.52, color=(220, 220, 220))
        y += 22

    # Thin accent line on right edge of panel
    cv2.line(frame, (panel_w, 0), (panel_w, h), color, 2)

    return frame


def _draw_text(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    font=cv2.FONT_HERSHEY_DUPLEX,
) -> None:
    # Drop shadow
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _draw_label_and_bar(
    frame: np.ndarray,
    label: str,
    value: float,
    pos: Tuple[int, int],
    w: int = 250,
    h_bar: int = 12,
    color: Tuple[int, int, int] = (100, 200, 255),
) -> None:
    x, y = pos
    _draw_text(frame, f"{label}: {value * 100:.0f}%", (x, y), scale=0.55)
    bar_y = y + 8
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + h_bar), (60, 60, 60), -1)
    fill = int(w * np.clip(value, 0, 1))
    cv2.rectangle(frame, (x, bar_y), (x + fill, bar_y + h_bar), color, -1)


def _score_color(score: float) -> Tuple[int, int, int]:
    """Green (good) → Orange → Red (bad)."""
    if score >= 0.8:
        return (50, 220, 50)
    elif score >= 0.5:
        return (50, 165, 255)
    else:
        return (50, 50, 220)


def _wrap_text(text: str, max_chars: int = 40) -> List[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_skeleton(frame: np.ndarray, landmarks, color: Tuple[int, int, int] = (0, 255, 128)) -> None:
    """
    Draw the MediaPipe Pose skeleton on the frame in-place.

    Args:
        frame:     BGR image
        landmarks: mediapipe NormalizedLandmarkList (results.pose_landmarks.landmark)
        color:     Line color (BGR)
    """
    h, w = frame.shape[:2]
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
        (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    ]
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 3, color, -1)
