import numpy as np
import json
from xor.utils import serialize_torch
import torch

# Build the dataset
def build_dataset():
    features = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float32)
    labels = np.array([0,1,1,0]).astype(np.float32).reshape(-1,1)
    
    result = []
    result["features"] = features
    result["labels"] = labels

    return result

def old_build_dataset():
    features = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float32)
    features = torch.from_numpy(features)
    serialized_features = serialize_torch("torch_features", features)
    
    labels = np.array([0,1,1,0]).astype(np.float32).reshape(-1,1)
    labels = torch.from_numpy(labels)
    serialized_labels = serialize_torch("torch_labels", labels)

    result = {}
    result["features"] = serialized_features
    result["labels"] = serialized_labels

    return result