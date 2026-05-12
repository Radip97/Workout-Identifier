"""
realtime_app.py — Real-Time Exercise Classification Application
================================================================
Captures webcam video, extracts MediaPipe Pose keypoints per frame,
maintains a sliding window, and runs the trained TF model for:
  • Exercise classification
  • Rep counting (angle threshold method)
  • Form evaluation (rule-based, per exercise)
  • Rich HUD overlay with all live metrics

Requirements:
  - Trained model saved via train.py (saved_models/)
  - Webcam connected

Usage:
  python realtime_app.py --model_dir saved_models
  python realtime_app.py --model_dir saved_models --camera 0 --seq_len 30
  python realtime_app.py --model_dir saved_models --record output.avi

Controls:
  q  — Quit
  r  — Reset rep counter
  s  — Screenshot
"""

import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

import mediapipe as mp

from inference import load_pipeline, Pipeline
from data_preprocessing import build_pose_extractor, extract_keypoints_from_frame
from utils import (
    normalize_keypoints,
    SlidingWindowBuffer,
    RepCounter,
    evaluate_form,
    draw_hud,
    draw_skeleton,
    compute_joint_angle,
    get_rep_angle_config,
    LM,
    NUM_LANDMARKS,
    FEATURES_PER_LANDMARK,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Inference is run every this many new frames (trade-off: latency vs CPU)
INFERENCE_EVERY_N_FRAMES = 5

# Confidence threshold below which classification is shown as "Detecting..."
CONFIDENCE_THRESHOLD = 0.55

# Minimum frames in buffer before first inference
MIN_BUFFER_FILL = 10   # partial window allowed for early display

# ---------------------------------------------------------------------------
# Real-time state machine
# ---------------------------------------------------------------------------

class ExerciseSession:
    """
    Manages per-session state: current exercise, rep counter, form eval buffer.
    Switches rep-counter configuration automatically when a new exercise is detected.
    """

    def __init__(self, seq_len: int, inference_every: int = INFERENCE_EVERY_N_FRAMES):
        self.seq_len          = seq_len
        self.inference_every  = inference_every

        self.buffer           = SlidingWindowBuffer(seq_len=seq_len)
        self.rep_counter      = RepCounter()
        self.current_exercise = None
        self.confidence       = 0.0
        self.form_score       = 0.5
        self.feedback         = "Waiting for pose..."
        self.is_perfect       = False
        self.current_angle    = None
        self.frame_count      = 0
        self.last_inference_t = 0.0

        # Form evaluation is run on every full window inference
        self._last_sequence: np.ndarray | None = None

    def update(
        self,
        frame_kps: np.ndarray,
        pipeline: Pipeline,
    ) -> None:
        """
        Add a new frame, run inference when due, update all state.

        Args:
            frame_kps: (132,) normalised keypoints for the current frame
            pipeline:  Loaded inference Pipeline
        """
        self.buffer.add(frame_kps)
        self.frame_count += 1

        # --- Rep counting (every frame based on detected exercise config) ---
        if self.current_exercise:
            cfg = get_rep_angle_config(self.current_exercise)
            
            if "joints" in cfg:
                angles = [
                    compute_joint_angle(frame_kps, p, j, d)
                    for p, j, d in cfg["joints"]
                ]
                angle = min(angles) if cfg.get("combine", "min") == "min" else sum(angles) / len(angles)
            else:
                angle = compute_joint_angle(
                    frame_kps,
                    cfg["proximal"], cfg["joint"], cfg["distal"],
                )
                
            self.current_angle = angle

            # If exercise changed, reset the rep counter
            self.rep_counter.update(angle)

        # --- Model inference ---
        if (
            self.buffer.is_ready()
            and self.frame_count % self.inference_every == 0
        ):
            sequence = self.buffer.get_sequence()          # (T, 132)
            self._last_sequence = sequence

            from inference import predict_sequence
            label, conf, form = predict_sequence(pipeline, sequence)

            if conf >= CONFIDENCE_THRESHOLD:
                # Detect exercise change → reset rep counter
                if label != self.current_exercise:
                    self.current_exercise = label
                    self.rep_counter.reset()
                self.confidence = conf
                self.form_score = form
            else:
                self.confidence = conf

            # Form feedback from rule-based evaluator
            if self.current_exercise and self._last_sequence is not None:
                try:
                    fscore, ftext, fperf = evaluate_form(
                        self.current_exercise, self._last_sequence
                    )
                    self.form_score  = fscore
                    self.feedback    = ftext
                    self.is_perfect  = fperf
                except Exception:
                    pass  # silently skip if form eval fails on partial data

    @property
    def display_exercise(self) -> str:
        if self.current_exercise is None or self.confidence < CONFIDENCE_THRESHOLD:
            return "Detecting..."
        return self.current_exercise

    @property
    def rep_count(self) -> int:
        return self.rep_counter.count


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def build_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float = 30.0,
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for optional recording."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------

def run_realtime(
    model_dir:    str  = "saved_models",
    camera_index: int  = 0,
    seq_len:      int  = None,          # None = use model config
    record_path:  str  = None,
    width:        int  = 1280,
    height:       int  = 720,
) -> None:
    """
    Main real-time classification loop.

    Args:
        model_dir:    Directory with trained model artefacts
        camera_index: OpenCV camera index (0 = default webcam)
        seq_len:      Override sequence length (uses model config if None)
        record_path:  Save output to video file (None = no recording)
        width/height: Desired capture resolution
    """
    # --- Load pipeline ---
    pipeline = load_pipeline(model_dir)
    effective_seq_len = seq_len or pipeline.seq_len
    print(f"Using sequence length: {effective_seq_len}")

    # --- Open webcam ---
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Connect a webcam or try a different --camera index."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera opened: {actual_w}×{actual_h} @ {actual_fps:.0f} fps")

    # --- Optional recording ---
    writer = None
    if record_path:
        writer = build_video_writer(record_path, actual_w, actual_h, actual_fps)
        print(f"Recording to: {record_path}")

    # --- MediaPipe pose ---
    pose = build_pose_extractor()

    # --- Session state ---
    session = ExerciseSession(seq_len=effective_seq_len)
    fps_counter = FPSCounter()

    print("\n=== Real-time classifier running ===")
    print("Controls:  [q] Quit   [r] Reset reps   [s] Screenshot")

    mp_drawing = mp.solutions.drawing_utils

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed. Retrying...")
                time.sleep(0.05)
                continue

            # Flip for mirror-like display
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # --- Extract keypoints ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                # Draw skeleton on display frame
                draw_skeleton(display, results.pose_landmarks.landmark)

                # Build flat keypoint vector
                kps = []
                for lm_pt in results.pose_landmarks.landmark:
                    kps.extend([lm_pt.x, lm_pt.y, lm_pt.z, lm_pt.visibility])
                kps_arr = np.array(kps, dtype=np.float32)
                norm_kps = normalize_keypoints(kps_arr)

                # Update session
                session.update(norm_kps, pipeline)
            else:
                # No pose — still advance buffer with zeros so window doesn't freeze
                zero_kps = np.zeros(NUM_LANDMARKS * FEATURES_PER_LANDMARK, dtype=np.float32)
                session.buffer.add(zero_kps)

            fps = fps_counter.tick()

            # --- Draw HUD ---
            display = draw_hud(
                frame=display,
                exercise=session.display_exercise,
                confidence=session.confidence,
                rep_count=session.rep_count,
                form_score=session.form_score,
                feedback=session.feedback,
                is_perfect=session.is_perfect,
                angle=session.current_angle,
            )

            # FPS counter (top-right)
            fps_text = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(display, fps_text,
                        (actual_w - tw - 15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

            # Buffer fill indicator
            buf_filled = len(session.buffer)
            if buf_filled < effective_seq_len:
                fill_text = f"Buffer: {buf_filled}/{effective_seq_len}"
                cv2.putText(display, fill_text,
                            (actual_w - 220, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)

            # --- Show and optionally record ---
            cv2.imshow("Exercise Classifier — Press Q to quit", display)

            if writer:
                writer.write(display)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("r"):
                session.rep_counter.reset()
                print("Rep counter reset.")
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ss_path = f"screenshot_{ts}.jpg"
                cv2.imwrite(ss_path, display)
                print(f"Screenshot saved: {ss_path}")

    finally:
        cap.release()
        pose.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Camera released. Session ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time exercise classification via webcam."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory with trained model artefacts (default: saved_models)"
    )
    # Legacy alias
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Alias: path like saved_models/model.h5; extracts model_dir automatically"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override sequence length (default: use model config)"
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Record output to this .avi file path"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Desired webcam width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Desired webcam height (default: 720)"
    )
    args = parser.parse_args()

    # Resolve model_dir
    model_dir = args.model_dir
    if args.model_path:
        model_dir = os.path.dirname(args.model_path) or model_dir

    run_realtime(
        model_dir=model_dir,
        camera_index=args.camera,
        seq_len=args.seq_len,
        record_path=args.record,
        width=args.width,
        height=args.height,
    )
