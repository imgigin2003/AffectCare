import json
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import os

from model import ElderCareModel
from prepare_data import prepare_data


def load_trained_model(model_path="best_model.pth"):
    # rebuild the empty architecture,
    # then pour the saved weights into it.
    model = ElderCareModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate(model, X_test, y_test, threshold=0.15):
    # Run the test set through the model once, get probabilities
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nFinal Evaluation (threshold={threshold})")
    print(f"Recall (caught emergencies):  {recall:.2%}")
    print(f"Precision (real alarms):      {precision:.2%}")
    print(f"F1 Score:                     {f1:.2%}")
    print(f"Confusion Matrix:\n{cm}")

    return recall, precision, f1, cm


def save_confusion_matrix(cm, output_path="../results/confusion-matrix.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,  # write the actual numbers inside each box
        fmt="d",  # format as integers, not decimals
        cmap="Blues",
        xticklabels=["Normal", "Distress"],
        yticklabels=["Normal", "Distress"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — AffectCare")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # frees memory, good practice when saving multiple plots
    print(f"Saved confusion matrix → {output_path}")


def save_loss_curve(
    loss_history_path="loss_history.json", output_path="../results/loss_curve.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(loss_history_path, "r") as f:
        loss_history = json.load(f)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker="o", markersize=3, color="#1D9E75")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Time — AffectCare")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curve → {output_path}")


if __name__ == "__main__":
    # Step 1 -> Load the same test split used during training
    # (same random_state=42, so this is the exact same test set, not new data)
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    X_test = X_test.reshape(-1, 1, 13, 173)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    # Step 2 -> Load the trained model
    model = load_trained_model()

    # Step 3 -> Run evaluation, print + save confusion matrix
    recall, precision, f1, cm = evaluate(model, X_test, y_test)
    save_confusion_matrix(cm)

    # Step 4 -> Save the loss curve from training
    save_loss_curve()
