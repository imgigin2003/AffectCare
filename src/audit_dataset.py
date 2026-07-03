import librosa
import numpy as np
import os


def flag_suspicious_files(folder_path, label_name, energy_threshold=0.15):
    """
    Flags files where energy is unusually high for their folder.
    A 'normal' file with scream-level energy is worth a manual listen.
    """
    suspicious = []

    for entry in os.scandir(folder_path):
        if not entry.name.endswith((".wav", ".mp3")):
            continue

        audio, sr = librosa.load(entry.path, sr=22050)
        rms_energy = np.mean(librosa.feature.rms(y=audio))

        if label_name == "normal" and rms_energy > energy_threshold:
            suspicious.append((entry.name, round(float(rms_energy), 4)))

    return suspicious


# Run this on your normal folder before training
flagged = flag_suspicious_files("../dataset/normal", "normal")
print(flagged)
# [('clip_017.wav', 0.21), ('clip_033.wav', 0.19)]  ← go listen to these manually
