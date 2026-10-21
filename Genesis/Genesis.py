import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

from Genesis import Transformer


class GenesisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 1)


    def forward(self):
        pass


def main():
    model = GenesisModel()
    




if __name__ == "__main__":
    main()
