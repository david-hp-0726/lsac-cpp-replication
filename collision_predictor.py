import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

def load_data():  
    X = np.load("data/X.npy")
    Y = np.load("data/Y.npy")

    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1))

# Model definition
class CollisionPredictor(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, x):
        return self.mlp(x)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5):
    # Evaluate before training 
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.6).float()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            val_loss += criterion(logits, yb).item() * xb.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"[Before Training] Val Loss: {avg_val_loss:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Start training 
    best_loss = float("inf")
    best_recall = -1
    best_f1 = -1
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.6).float()
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                val_loss += criterion(logits, yb).item() * xb.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_recall = recall
            best_f1 = f1
            torch.save(model.state_dict(), "model/best_collision_predictor.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    print(f"Best Val Loss: {best_loss:.4f} | Best Recall: {best_recall:.4f} | Best F1: {best_f1:.4f}")


def main():
    X_train, X_val, Y_train, Y_val = load_data()
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=1024)

    model = CollisionPredictor()
    positive_count = (Y_train.sum() + Y_val.sum()).item()
    negative_count = len(Y_train) + len(Y_val) - positive_count
    pos_weight = torch.tensor(negative_count / positive_count)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train(model, train_loader, val_loader, criterion, optimizer)

if __name__ == "__main__":
    main()
