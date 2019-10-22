import os
from PIL import Image
import numpy as np
import torch
import torchvision as tv

class CASIADataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        