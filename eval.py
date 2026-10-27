import os
import torch
import torch.nn as nn
from dataset.dataset import prepare_data
from models.CustomNet import CustomNet
from utils.engine import evaluate


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  _, val_loader = prepare_data()

  model = CustomNet(num_classes=200).to(device)
  ckpt_path = os.path.join("checkpoints", "best_model.pth")

  if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Pesi caricati correttamente da {ckpt_path}")
  else:
    print("Nessun checkpoint trovato! Esegui prima il training.")
    return

  criterion = nn.CrossEntropyLoss()
  val_loss, val_acc = evaluate(model, val_loader, criterion, device)
  print(f"\nRisultati sul Validation Set:")
  print(f"Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}%")


if __name__ == "__main__":
  main()