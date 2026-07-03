import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

import json
import pickle
import copy

from model import ElderCareModel
from prepare_data import prepare_data
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import StandardScaler


def train_model(epochs=40, batch_size=32, lr=0.0005):
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Step 1 -> Reshape back into images
    X_train = X_train.reshape(-1, 1, 13, 173)
    X_test = X_test.reshape(-1, 1, 13, 173)
    # -1 in the first dimension means "infer this automatically based on the other dimensions.
    # " Since we know the total size is (N, 2249) and we're specifying (1, 13, 173) for the rest,
    # NumPy calculates that the first dimension must be N.

    # Step 2 -> Convert to Pytorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).float().unsqueeze(1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    # Step 3 -> Create data loaders
    # In PyTorch, we feed data to the model in batches using DataLoader. This handles shuffling,
    # batching, and parallelization for us.
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Step 4 -> Now we can create an instance of our ElderCareModel
    model = ElderCareModel()

    # Step 5 -> loss, optimizer
    pos_weight = torch.tensor([1.5])  # missing distress costs 2x a false alarm
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    loss_history = []
    best_f1 = 0
    best_model_state = None

    # Step 6 -> Training Loop
    # Since you're using DataLoader, the loop has one more layer — you loop over batches, not the whole dataset at once
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                test_probs = torch.sigmoid(test_logits)
                test_preds = (test_probs >= 0.15).float()
                current_recall = recall_score(y_test, test_preds)
                current_f1 = f1_score(
                    y_test, test_preds
                )  # balances recall AND false alarms

            cm = confusion_matrix(y_test, test_preds)
            false_alarms = cm[0][1]
            print(
                f"  → Epoch {epoch+1}: recall {current_recall:.2%}, F1 {current_f1:.2%}, false alarms {false_alarms}"
            )

            if current_f1 > best_f1:  # save on F1, not recall alone
                best_f1 = current_f1
                best_model_state = copy.deepcopy(model.state_dict())

            model.train()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, X_test, y_test, scaler, loss_history


def evaluate_model(model, X_test, y_test, threshold=0.2):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits)

    # Evaluate with the threshold passed into this function
    predictions = (probabilities >= threshold).float()
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nRecall (caught emergencies): {recall:.2%}")
    print(f"Precision (no false alarms): {precision:.2%}")
    print("Confusion Matrix:")
    print(cm)

    return recall, cm


if __name__ == "__main__":
    model, X_test, y_test, scaler, loss_history = train_model()
    evaluate_model(model, X_test, y_test, threshold=0.15)

    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f)

    # Save the trained model weights
    # best_model.pth — the model's weights, frozen at the best F1 checkpoint.
    torch.save(model.state_dict(), "best_model.pth")

    # scaler.pkl — the fitted StandardScaler. Any new audio file must be scaled with this exact scaler, not a fresh one
    # or the numbers won't be in the same range the model learned from.
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model and scaler saved.")
