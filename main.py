import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import Genesis.Genesis as gen

GenesisModel = gen.GenesisModel()

IMG_SIZE = 32

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),                                  # PIL -> [3, H, W], wartości 0..1
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # -> mniej więcej -1..1
])


def apply_transform(batch):
    # set_transform dostaje batch jako słownik list
    batch["pixel_values"] = [transform(img.convert("RGB")) for img in batch["img"]]
    del batch["img"]
    return batch


def main():
    ds = load_dataset("uoft-cs/cifar10")    # DatasetDict: train / test
    print(ds)
    print(ds["train"].features)

    ds.set_transform(apply_transform)       # zamiast with_format("torch")

    train_loader = DataLoader(ds["train"], batch_size=32, shuffle=True, num_workers=2)

    batch = next(iter(train_loader))
    print(batch["pixel_values"].shape)      # torch.Size([32, 3, 32, 32])
    print(batch["label"].shape)             # torch.Size([32])


if __name__ == "__main__":
    main()