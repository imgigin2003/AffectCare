# preprocessing.py
# Librosa
# using this library helps us load and process audio files.

# .load() -> get the raw signal into memory
# .trim() -> cut the silence while the audio is still raw
# .fix_length() -> pad/cut to a standard size
# .mfcc() -> measures only real, standardized audio

import librosa
import os
import numpy as np


def load_and_format(filepath, sr=22050, target_seconds=4, n_mfcc=13):
    # Step 1 -> Load the audio file
    # 22050 is the CD-Quality standard. enough to capture all human vocal frequencies.
    # mono=True collapses stereo (left+right) into one channel — we don't need stereo for distress detection
    audio, sr = librosa.load(filepath, sr=sr, mono=True)
    # Why 22050? Human screams and groans live below 8000 Hz. By the Nyquist theorem, you need to
    # sample at twice that frequency minimum. 22050 is the comfortable standard — high enough to
    # capture everything, low enough to keep files manageable.

    # Step 2 -> Trim the audio file
    # trimming it returns a cleaned audio AND the indices it cut at
    # top_db=18 means anything quieter than 18 decibels gets removed.
    # we might have a whispered groan, so we set this with caution.
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=18)
    # The _ variable: The second return value is the trim indices [start, end].
    # We discard it with _ because we only need the cleaned audio itself.

    # Step 3 -> Fix the Length
    # fix_length pads with zeros if its too short, truncates it if its too long
    target_length = target_seconds * sr
    audio_fixed = librosa.util.fix_length(audio_trimmed, size=target_length)
    # Why 4 seconds? A fall event — the thud, the gasp, the silence after —
    # typically completes within 3-4 seconds. Long enough to capture the full event,
    # short enough to keep the model lightweight.

    # Step 4 -> Extract the MFCCs
    # n_mfcc=13: the 13 most perceptually significant frequency bands
    # The output shape is (13, T) where T = number of time frames
    mfccs = librosa.feature.mfcc(y=audio_fixed, sr=sr, n_mfcc=n_mfcc)
    # This is the moment raw sound becomes a picture. Each of the 13 rows is a frequency band.
    # Each column is a moment in time. Each value is the energy level.
    # Your CNN will read this exactly like an image.

    # Step 5 -> Add Channel Dimension (shape -> (1, 13, 173))
    mfccs = np.expand_dims(mfccs, axis=0)

    return mfccs


def build_dataset(dataset_path, sr=22050, target_seconds=4, n_mfcc=13):
    X = []  # features — each item will be shape (1, 13, 173)
    y = []  # labels  — 1 for distress, 0 for normal
    supported_formats = [".mp3", ".wav"]

    # folder name IS the label — no need to read the file to know
    label_map = {"distress": 1, "normal": 0}

    for class_name, label in label_map.items():
        class_folder = os.path.join(dataset_path, class_name)
        for entry in os.scandir(class_folder):
            # Skip anything that isn't an audio file
            if not any(entry.name.endswith(fmt) for fmt in supported_formats):
                continue
            features = load_and_format(entry.path, sr, target_seconds, n_mfcc)
            X.append(features)  # shape: (1, 13, 173)
            y.append(label)  # 1 or 0

    # Stack everything into two clean arrays
    X = np.array(X)  # shape: (num_files, 1, 13, 173)
    y = np.array(y)  # shape: (num_files,)

    print(f"Dataset built: {X.shape[0]} files")
    print(f"Feature shape: {X.shape}")
    print(f"Distress samples: {y.sum()}")
    print(f"Normal samples:   {(y == 0).sum()}")

    return X, y


if __name__ == "__main__":
    # Run this ONCE to extract MFCCs from all raw audio and save to disk.
    # After this, training reads from data/processed/ instead of re-processing every run.
    # This drops training time from ~45 seconds to ~5 seconds per run.
    print("Extracting MFCCs from raw audio — this runs once...")

    X, y = build_dataset("../dataset")

    os.makedirs("../data/processed", exist_ok=True)
    np.save("../data/processed/X.npy", X)
    np.save("../data/processed/y.npy", y)

    print(f"Saved → data/processed/X.npy  {X.shape}")
    print(f"Saved → data/processed/y.npy  {y.shape}")
