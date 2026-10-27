import os
import torch
from torchvision import datasets, transforms


def prepare_data(data_dir="./data/tiny-imagenet-200", batch_size=64):
  transform_train = transforms.Compose(
      [
          transforms.RandomCrop(64, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
          ),
      ]
  )

  transform_val = transforms.Compose(
      [
          transforms.ToTensor(),
          transforms.Normalize(
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
          ),
      ]
  )

  train_dir = os.path.join(data_dir, "train")
  val_dir = os.path.join(data_dir, "val")

  train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
  val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
  )
  val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
  )

  return train_loader, val_loader