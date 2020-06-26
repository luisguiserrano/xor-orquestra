import numpy as np
import json
from xor.utils import save_torch
import torch

# Build the dataset
def build_dataset():
    features = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float32)
    features = torch.from_numpy(features)

    labels = np.array([0,1,1,0]).astype(np.float32).reshape(-1,1)
    labels = torch.from_numpy(labels)

    return features, labels