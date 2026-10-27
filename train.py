import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import prepare_data
from models.CustomNet import CustomNet
from utils.engine import evaluate, train_one_epoch


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Dispositivo in uso: {device}")

  train_loader, val_loader = prepare_data()

  model = CustomNet(num_classes=200).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  os.makedirs("checkpoints", exist_ok=True)
  best_val_acc = 0.0
  num_epochs = 10

  for epoch in range(num_epochs):
    print(f"\n--- Epoca {epoch + 1}/{num_epochs} ---")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
    )
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      ckpt_path = os.path.join("checkpoints", "best_model.pth")
      torch.save(model.state_dict(), ckpt_path)
      print(f"-> Modello salvato in {ckpt_path}")


if __name__ == "__main__":
  main()