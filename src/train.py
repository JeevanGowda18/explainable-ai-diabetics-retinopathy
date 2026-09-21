import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Corrected Import: dataset.py lives inside the 'data' folder
from data.dataset import DiabeticRetinopathyDataset, get_data_transforms
from src.model import DRClassifier

def train_model(data_dir, csv_file, epochs=5, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms & Datasets
    train_transform, _ = get_data_transforms()
    dataset = DiabeticRetinopathyDataset(csv_file=csv_file, img_dir=data_dir, transform=train_transform)
    
    # DataLoader with multi-processing & pinned memory for GPU speedup
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2 if os.name != 'nt' else 0, # Set to 0 on Windows to avoid spawn crashes
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Model, Loss, Optimizer
    model = DRClassifier(num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    # Save model weights to the root 'models' folder
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "dr_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")

if __name__ == "__main__":
    # Example execution (update paths to match your local dataset location)
    DATA_DIR = "data/train_images"
    CSV_FILE = "data/train.csv"
    
    if os.path.exists(CSV_FILE) and os.path.exists(DATA_DIR):
        train_model(data_dir=DATA_DIR, csv_file=CSV_FILE, epochs=5, batch_size=16)
    else:
        print("Train module ready. Provide valid paths to APTOS 2019 dataset CSV and image directory to start training.")