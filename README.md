# Real-Time Workout Classifier

This project implements a real-time workout classifier using MediaPipe pose estimation and a deep learning model to identify different exercises from a video feed.

## Features

- **Pose Detection**: Uses MediaPipe to extract 33 pose keypoints from video frames.
- **Deep Learning Model**: Utilizes a Bidirectional LSTM with Temporal Attention (TensorFlow/Keras) for time-series classification of exercise movements.
- **Exercise Classification**: Classifies 7 different workout exercises:
  - Bench Press
  - Bodyweight Squats
  - Lunges
  - Pull-ups
  - Push-ups
  - Barbell Biceps Curl
  - Hammer Curl
- **Real-time Classification**: Processes webcam feed in real-time.
- **Form Evaluation**: Evaluates form quality with a dedicated regression head and rule-based feedback.
- **Rep Counting**: Automatically counts repetitions based on joint angle thresholds.
- **Rich HUD**: Real-time overlay showing current exercise, confidence, rep count, form score, and joint angles.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data** (Extract keypoint sequences from video dataset):
   ```bash
   python data_preprocessing.py
   ```
   *This requires a video dataset in the `workout/` directory.*

3. **Train the Model**:
   ```bash
   python train.py --data_dir processed_data --model_dir saved_models
   ```

4. **Run Real-time Classification**:
   ```bash
   python realtime_app.py --model_dir saved_models
   ```

## Project Structure

```text
├── data_preprocessing.py     # Extracts pose keypoints into sequence data
├── model.py                  # TensorFlow/Keras BiLSTM + Attention model architecture
├── train.py                  # Model training pipeline
├── inference.py              # Inference pipeline for the trained model
├── utils.py                  # Rep counting, form evaluation, HUD drawing utilities
├── realtime_app.py           # Real-time webcam application
├── requirements.txt          # Python dependencies
├── saved_models/             # Directory containing trained model and artefacts
│   ├── model.h5
│   ├── scaler.pkl
│   └── label mappings
├── processed_data/           # Directory for extracted sequence data
└── workout/                  # Video dataset directory
```

## How It Works

1. **Feature Extraction**: MediaPipe extracts 33 pose landmarks (x, y, z coordinates + visibility) from each video frame.
2. **Preprocessing**: The `data_preprocessing.py` script normalizes these keypoints and structures them into sliding windows (sequences).
3. **Training**: The `train.py` script trains a Bidirectional LSTM model with temporal attention to classify exercises and evaluate form based on the sequences.
4. **Real-time Inference**: The `realtime_app.py` script captures the webcam feed, maintains a sliding window of recent frames, and passes them to the model for real-time classification, form feedback, and rep counting.

## Usage

- **Training**: Run `train.py` after preprocessing your video data. You can configure hyperparameters in `model.py` or pass arguments (e.g., `--epochs 100`).
- **Classification**: Run `realtime_app.py` to start the live camera feed.
- **Controls**: 
  - Press `q` to quit the real-time classification window.
  - Press `r` to reset the rep counter.
  - Press `s` to save a screenshot of the current frame.

## Requirements

- Python 3.8+
- Webcam for real-time classification
- Video dataset in the `workout/` directory for training.

## Model Performance

The training script generates loss/accuracy curves (`training_curves.png`) and a confusion matrix (`confusion_matrix.png`) in the `saved_models/` directory for evaluating model performance.

## Customization

You can modify the model's configuration via the `CONFIG` dictionary in `model.py`. To add new exercises, ensure your `workout/` directory contains folders with the new exercise videos and rerun the preprocessing and training scripts.