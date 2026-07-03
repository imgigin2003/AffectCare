import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
from model import ElderCareModel
from preprocessing import load_and_format


def load_trained_model(model_path="src/best_model.pth"):
    # Step 1 -> Rebuild empty architecture
    model = ElderCareModel()

    # Step 2 -> Load the saved weights
    model.load_state_dict(torch.load(model_path))

    # Step 3 -> Switch to evaluation mode
    model.eval()

    return model


def load_scaler(scaler_path="src/scaler.pkl"):
    # The scaler learned its "normal range" from your training data.
    # Any new audio MUST be scaled with this exact same scaler,
    # not a fresh one, or the numbers won't mean the same thing to the model.

    # Step 4 -> Loading the scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def predict(filepath, model, scaler, threshold=0.15):
    # Step 5 -> Run the exact same preprocessing pipeline used in training
    # (load, trim, fix_length, mfcc) — this is the same function from preprocessing.py
    features = load_and_format(filepath)

    # Step 6 -> Flatten to match what the scaler expects
    # The scaler was fit on flattened (N, 2249) arrays during training
    feature_flat = features.reshape(1, -1)

    # Step 7 -> Scale using the SAME scaler from training
    feature_scaled = scaler.transform(feature_flat)

    # Step 8 -> Reshape back into image form for the CNN
    features_image = feature_scaled.reshape(1, 1, 13, 173)

    # Step 9 -> Convert to a PyTorch tensor
    input_tensor = torch.from_numpy(features_image).float()

    # Step 10 -> Run it through the model
    with torch.no_grad():  # no gradient tracking needed — we're not training
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()  # pulls out the plain number

    # Step 11 -> Apply the same Recall-First threshold used during training
    label = "distress" if probability >= threshold else "normal"

    return label, probability


if __name__ == "__main__":
    # Step 12 -> Read the file path from the command line
    # sys.argv[0] is always the script name itself ("predict.py")
    # sys.argv[1] is the first argument the user typed after it
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_audio_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Step 13 -> Load the trained model and scaler from disk
    model = load_trained_model()
    scaler = load_scaler()

    # Step 14 -> Run the prediction
    label, probability = predict(filepath, model, scaler)

    # Step 15 -> Print the result, matching the plan's exact desired format
    print(f"→ Prediction: {label} (confidence: {probability * 100 :.2f})%")
